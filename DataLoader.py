# from torch.utils.data import Dataset
# from pathlib import Path
import pathlib
import matplotlib.pyplot as plt

import cv2
import timeit
import numpy as np
import os
import torchvision
import torch
import matplotlib.pyplot as plt
from utils import getNumRecordings, getListFromFile, getNumFrames, saveVideoClip, addTextToImg, read16BitPGM, imread_pgm
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plyfile
class HololensStreamRecBase():
    """Face Landmarks dataset."""

    def __init__(self, dataset_path, furniture_list: list, action_list_filename='action_list.txt',
                 train_filename='all_train_dir_list.txt', test_filename='all_train_dir_list.txt', transform=None,
                 gt_annotation_filename='db_gt_annotations.json'):
        """
        Args:
            action_list_filename (string): Path to the csv file with annotations.
            dataset_path (string): Root directory with all the data.
            furniture_list = list of strings containing names of furniture assembled in dataset.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """

        self.dataset_root_path = dataset_path
        self.furniture_dirs = [os.path.join(dataset_path, furniture_name) for furniture_name in furniture_list]

        for furniture_dir in self.furniture_dirs:
            assert os.path.exists(furniture_dir)

        self.furniture_dir_sizes = [getNumRecordings(_dir_) for _dir_ in self.furniture_dirs]
        self.num_recordings = sum(self.furniture_dir_sizes)

        # indexing_files:
        self.gt_annotation_filename = os.path.join(dataset_path, 'indexing_files', gt_annotation_filename)
        self.action_list_filename = os.path.join(dataset_path, 'indexing_files', action_list_filename)
        self.train_filename = os.path.join(dataset_path, 'indexing_files', train_filename)
        self.test_filename = os.path.join(dataset_path, 'indexing_files', test_filename)

        # load lists from files:
        self.action_list = getListFromFile(self.action_list_filename)
        self.action_list.sort()
        if "N/A" in self.action_list:
            self.action_list.remove("N/A")

        self.action_list.insert(0, "N/A")  # 0 label for unlabled frames
        self.num_classes = len(self.action_list)
        self.train_video_list = getListFromFile(self.train_filename)
        self.test_video_list = getListFromFile(self.test_filename)
        self.all_video_list = self.test_video_list + self.train_video_list
        self.action_name_to_id_mapping = {}
        self.id_to_action_name_mapping = {}
        for action_id, action in enumerate(self.action_list):
            self.action_name_to_id_mapping[action] = action_id
            self.id_to_action_name_mapping[action_id] = action
        print(self.action_name_to_id_mapping)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    def __len__(self):
        return

    def get_video_info_table(self, dataset='all'):
        """
        fetch the annotated videos table from the database
        :param : device: string ['all', 'dev1', 'dev2', 'dev3']
        :return: annotated videos table
        """
        video_data_table = []
        if dataset == "all":
            rec_list = self.all_video_list
        elif dataset == "train":
            rec_list = self.train_video_list
        elif dataset == "test":
            rec_list = self.test_video_list
        else:
            raise ValueError("Invalid dataset name")

        print("using json annotation file {}".format(self.gt_annotation_filename))
        # with open(self.gt_annotation_filename) as json_file_obj:
        #     db_gt_annotations = json.load(json_file_obj)

        for _dir_ in rec_list:
            row = {"nframes": getNumFrames(_dir_), 'video_path': _dir_}

            # if dataset != all:
            # print(dataset, db_gt_annotations["database"][_dir_]["subset"])
            # assert dataset in db_gt_annotations["database"][_dir_]["subset"]
            video_data_table.append(row)
        return video_data_table

    def get_video_annotations_table(self, video_path):
        with open(self.gt_annotation_filename) as json_file_obj:
            db_gt_annotations = json.load(json_file_obj)
        # print(db_gt_annotations.keys())
        # print(db_gt_annotations["database"].keys())

        if video_path in db_gt_annotations["database"].keys():
            return db_gt_annotations["database"][video_path]["annotation"]
        else:
            return None

    def get_video_table(self, video_idx):
        """
        fetch the video information row from the video table in the database
        :param :  video_idx: index of the desired video
        :return: video information table row from the databse
        """
        return self.cursor_annotations.execute('''SELECT * FROM videos WHERE id = ?''', (video_idx,))

    def get_annotation_table(self):
        """
        :return: full annotations table (for all videos)
        """
        return self.cursor_annotations.execute('''SELECT * FROM annotations ''')


class HololensStreamRecClipDataset(HololensStreamRecBase):
    def __init__(self, dataset_path, furniture_list: list, action_list_filename='action_list.txt',
                 train_filename='all_train_dir_list.txt', test_filename='all_train_dir_list.txt', transform=None,
                 gt_annotation_filename='db_gt_annotations.json', modalities=["all"], frame_skip=1, frames_per_clip=32,
                 dataset="train", rgb_label_watermark=False, furniture_mod = ["all"]):

        super().__init__(dataset_path, furniture_list, action_list_filename,
                         train_filename, test_filename, transform, gt_annotation_filename)

        # self.camera = camera
        # self.resize = resize
        # self.input_type = input_type
        self.rgb_label_watermark = rgb_label_watermark
        self.modalities = modalities
        self.transform = transform
        self.set = dataset
        self.frame_skip = frame_skip
        self.frames_per_clip = frames_per_clip

        if self.set == 'train':
            self.video_list = self.filterFurnitureModalities(self.train_video_list)
        elif self.set == 'test':
            self.video_list = self.filterFurnitureModalities(self.test_video_list)
        else:
            raise ValueError("Invalid set name")
        print("got the following video list: ", self.video_list)
        self.annotated_videos = self.get_video_frame_labels()
        self.clip_set, self.clip_label_count = self.get_clips()
        # print(self.clip_set)
        labels =[]
        clip_labels_count =[]
        for i, label in enumerate(self.action_list):
            # print((label, self.clip_label_count[i]))
            if(self.clip_label_count[i]>20000):
                continue
            labels.append(label)
            clip_labels_count.append(self.clip_label_count[i])

        # Create a dataframe with the bin names and values
        df = pd.DataFrame({'bin': labels, 'count': clip_labels_count})
        # Create a histogram figure
        fig = px.histogram(df, x='bin', y='count', title='Histogram')
        fig.write_html("D:\loaded_clips\LabeledFramesPerAction.html")

    def filterFurnitureModalities(self, rec_list, furniture_mod):
        if furniture_mod == ["all"]:
            return rec_list
        filtered_rec_list = []
        for furniture_name in furniture_mod:
            for rec in rec_list:
                if f"\HoloLens\\{furniture_name}\\" in rec:
                    filtered_rec_list.append(rec)
        return filtered_rec_list


    def get_video_frame_labels(self):
        # Extract the label data from the database
        # outputs a dataset structure of (video_path, multi-label per-frame, number of frames in the video)
        video_info_table = self.get_video_info_table()
        vid_list = []
        for row in video_info_table:
            n_frames = int(row["nframes"])
            video_path = row['video_path']
            # video_name = os.path.join(video_path.split('/')[0], video_path.split('/')[1])

            rec_frame_labels = np.zeros((self.num_classes, n_frames), np.float32)  # allow multi-class representation
            rec_frame_labels[0, :] = np.ones((1, n_frames),
                                             np.float32)  # initialize all frames as NA
            # video_id = row['id']
            annotation_table = self.get_video_annotations_table(video_path)
            if not annotation_table:
                # print(f"reached an unannotated directory: {video_path}!!!")
                continue
            for ann_row in annotation_table:
                action = ann_row["label"]  # map the labels

                if action == 'pick up small coffee table screw': action = 'pick up small screw'
                if action == 'allign small coffee table screw': action = 'allign small screw'
                if action == 'vr interface interaction': action = 'application interface'
                if action == 'spin screw': action = 'spin screwdriver'
                if action == 'pick up back panel ': action = 'pick up back panel'
                if action == 'application interface ': action = 'application interface'
                if action == 'pick up side panel' : action = 'pick drawer up side panel'
                if action == 'pick up drawer side panel' : action = 'pick up side panel'
                if action == 'pick up screw holder (the strange white thing)' : action = 'pick up cam lock'
                if action == 'insert screw holder (the strange white thing)' : action = 'insert cam lock'
                if action == 'lay down screwdriver ' : action = 'lay down screwdriver'
                if action == 'pick up drawer bottom panel ' : action = 'pick up bottom panel'
                if action == 'allign drawer bottom panel '  or action == 'allign drawer back panel': action = 'N/A' #TODO: fix this!!!
                if action == 'spin drawer knob screw' : action = 'spin drawer knob'
                if action == 'pick drawer up side panel' : action =  'pick up side panel'

                action_id = self.action_name_to_id_mapping[action]
                # object_id = ann_row["object_id"]
                # action_id = self.get_action_id(atomic_action_id, object_id)
                start_frame = ann_row['segment'][0]
                end_frame = ann_row['segment'][1]
                end_frame = end_frame if end_frame < n_frames else n_frames
                if action is not None:
                    rec_frame_labels[:, start_frame:end_frame] = 0  # remove the N/A
                    rec_frame_labels[action_id, start_frame:end_frame] = 1
                    # print(rec_frame_labels)
            vid_list.append(
                (video_path, rec_frame_labels, n_frames))  # 0 = duration - irrelevant for initial tests, used for start
        return vid_list

    def get_clips(self):
        # extract equal length video clip segments from the full video dataset
        clip_dataset = []
        label_count = np.zeros(self.num_classes)
        # for i, data in enumerate(self.annotated_videos):
        for i, data in enumerate(self.annotated_videos):
            n_frames = data[2]
            n_clips = int(n_frames / (self.frames_per_clip * self.frame_skip))
            # remaining_frames = n_frames % (self.frames_per_clip * self.frame_skip)
            for j in range(0, n_clips):
                for k in range(0, self.frame_skip):
                    start = j * self.frames_per_clip * self.frame_skip + k
                    end = (j + 1) * self.frames_per_clip * self.frame_skip
                    label = data[1][:, start:end:self.frame_skip]

                    label_count = label_count + np.sum(label, axis=1)
                    frame_ind = np.arange(start, end, self.frame_skip).tolist()
                    clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, 0))

        return clip_dataset, label_count

    def getLabelsInClipIdx(self, np_labels):
        print(np_labels.size)
        action_strings = [self.id_to_action_name_mapping[np.argmax(np_labels[i])] for i in range(len(np_labels))]
        print(action_strings)
        return action_strings


    def load_point_clouds_2(self, rec_dir, frame_indices):
        frames = []
        for counter in frame_indices:
            target_points = torch.zeros(92160, 9)
            point_cloud_full_path = os.path.join(rec_dir, "norm", "Depth Long Throw", "{}.ply".format(i))
            plydata = plyfile.PlyData.read(point_cloud_full_path)
            d = np.asarray(plydata['vertex'].data)
            pc = np.column_stack([d[p.name] for p in plydata['vertex'].properties])
            target_points[:pc.shape[0], :] = torch.from_numpy(pc)
            frames.append(target_points)
        return torch.stack(frames, 0)


    def load_point_clouds(self, rec_dir, frame_indices):
        point_clouds = []
        for index in frame_indices:
            point_cloud_full_path = os.path.join(rec_dir, "norm", "Depth Long Throw", "{}.ply".format(index))
            target_points = torch.zeros(92160, 9)
            ply_data = plyfile.PlyData.read(point_cloud_full_path)
            points = ply_data['vertex'].data
            # [(),(),()]=>[[],[],[]]
            points = [list(point) for point in points]
            target_points[:len(points), :] = torch.tensor(points)
            point_clouds.append(target_points)
        return torch.stack(point_clouds)

    def load_data_frames_from_csv(self, rec_dir, frame_indices, filename):
        full_rec_csv_path = os.path.join(rec_dir, "norm", filename)

        with open(full_rec_csv_path, "rb") as full_rec_csv_f:
            clip_data = np.loadtxt(full_rec_csv_f, delimiter=",")[frame_indices, :]
        return torch.Tensor(clip_data)

    def load_depth_frames(self, rec_dir, frame_indices):
        depth_frames = []
        for index in frame_indices:
            depth_frame_full_path = os.path.join(rec_dir, "norm", "Depth Long Throw", "{}.pgm".format(index))
            pgm_data = imread_pgm(depth_frame_full_path)
            # pgm_data = read16BitPGM(depth_frame_full_path)
            depth_frames.append(pgm_data)
        return torch.Tensor(depth_frames)

    def load_rgb_frames(self, rec_dir, frame_indices, labels=[]):
        # load video file and extract the frames
        np_labels = np.array(labels).T
        frames = []
        print(frame_indices)
        # TODO: only get labels when watermark is necessary
        str_labels = self.getLabelsInClipIdx(np_labels)
        assert (self.rgb_label_watermark and len(labels) > 0) or not self.rgb_label_watermark
        for frame_num in frame_indices:
            rgb_frame_full_path = os.path.join(rec_dir, "norm", "pv", "{}.png".format(frame_num))
            assert os.path.exists(rgb_frame_full_path)
            if self.rgb_label_watermark:
                frame = addTextToImg(rgb_frame_full_path, str_labels[frame_num - frame_indices[0]] + f", {frame_num}")
                # print(frame.shape)
            else:
                frame = torch.Tensor(torchvision.io.read_image(rgb_frame_full_path))
            frames.append(frame)

        print(len(frames), len(frames[0]), len(frames[0][0]), len(frames[0][0][0]))
        frames = torch.stack(frames)
        print(frames.shape)
        return frames

    #
    # def video_to_tensor(self, pic):
    #     """Convert a ``numpy.ndarray`` to tensor.
    #     Converts a numpy.ndarray (T x H x W x C)
    #     to a torch.FloatTensor of shape (C x T x H x W)
    #
    #     Args:
    #          pic (numpy.ndarray): Video to be converted to tensor.
    #     Returns:
    #          Tensor: Converted video.
    #     """
    #     return torch.tensor(pic.transpose([3, 0, 1, 2]), dtype=torch.float32)

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.clip_set)

    def __getitem__(self, index):
        # 'Generate one sample of data'
        recording_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]
        # return video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad
        print(f"getting clip from recording {recording_full_path}")
        clip_modalities_dict = {}
        if self.modalities == ["all"]:
            print("returning all modalities")
            #TODO: remove labels argument from load_rgb_frames
            clip_modalities_dict["rgb_frames"] = self.load_rgb_frames(recording_full_path, frame_ind, labels)
            clip_modalities_dict["depth_frames"] = self.load_depth_frames(recording_full_path, frame_ind)
            clip_modalities_dict["point_clouds"] = self.load_point_clouds_2(recording_full_path, frame_ind)
            clip_modalities_dict["eye_data_frames"] = self.load_data_frames_from_csv(recording_full_path, frame_ind,
                                                                                     filename="norm_proc_eye_data.csv")
            clip_modalities_dict["hand_data_frames"] = self.load_data_frames_from_csv(recording_full_path, frame_ind,
                                                                                     filename="norm_proc_hand_data.csv")

            return clip_modalities_dict, torch.from_numpy(labels),  vid_idx, frame_pad

        for mod in self.modalities:
            if mod == "pv":
                clip_modalities_dict["rgb_frames"] = self.load_rgb_frames(recording_full_path, frame_ind)
            elif mod == "point_clouds":
                clip_modalities_dict["point_clouds"] = self.load_point_clouds_2(recording_full_path, frame_ind)
            elif mod == "depth_frames":
                clip_modalities_dict["depth_frames"] = self.load_depth_frames(recording_full_path, frame_ind)
            elif mod == "eye_data_frames":
                clip_modalities_dict["eye_data_frames"] = self.load_data_frames_from_csv(
                    recording_full_path, frame_ind, filename="norm_proc_eye_data.csv")
            elif mod == "hand_data_frames":
                clip_modalities_dict["hand_data_frames"] = self.load_data_frames_from_csv(
                    recording_full_path, frame_ind, filename="norm_proc_hand_data.csv")

        return clip_modalities_dict, torch.from_numpy(labels),  vid_idx, frame_pad

        # return self.video_to_tensor(rgb_clip), torch.from_numpy(labels), vid_idx, frame_pad




if __name__ == "__main__":
    dataset_path = r'C:\HoloLens'
    furniture_list = ["Coffee_Table"]
    frames_per_clip_list = [8,16,32,64]
    # frames_per_clip_list = [8]#,16,32,64]
    run_times = [0,0,0,0]
    # run_times = [0]#,0,0,0]
    clip_num = 0
    num_runs=1
    dataset = HololensStreamRecClipDataset(dataset_path, furniture_list, frames_per_clip=32,
                                           rgb_label_watermark=False, modalities=["rgb_frames"])
    exit()
    for run in range(num_runs):
        for i, frames_per_clip in enumerate(frames_per_clip_list):
            dataset = HololensStreamRecClipDataset(dataset_path, furniture_list, frames_per_clip=frames_per_clip,
                                                   rgb_label_watermark=False, modalities=["depth_frames"])
            start = timeit.default_timer()
            clip_data_dict, labels, vid_idx, frame_pad = dataset[clip_num]
            stop = timeit.default_timer()
            run_times[i] += (stop-start)/num_runs
            print(stop-start)
    print(run_times)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.plot(frames_per_clip_list, run_times)
    plt.xlabel('clip length')
    plt.ylabel('time [ms]')
    plt.title('getitem time for eye and hand data')
    for xy in zip(frames_per_clip_list, run_times):  #
        ax.annotate(f'({xy[0]:.3f},{1000*xy[1]:.3f})', xy=xy, textcoords='data')

    plt.show()
    exit()
    clip_num = 0
    clip_data_dict, labels,  vid_idx, frame_pad = dataset[clip_num]
    clip_frames = clip_data_dict["rgb_frames"]
    for mod in clip_data_dict.keys():
        print(mod)
        print(clip_data_dict[mod].shape)

    # vid_clip_name = r"D:\loaded_clips\clip_{}.avi".format(clip_num)
    # saveVideoClip(vid_clip_name, clip_frames)
    # user_input = input("input frame number")
    # while(user_input!="q"):
    #
    #     user_input = int(user_input)
    #     if (user_input >= len(dataset)):
    #         print("you idiot")
    #     clip_frames = dataset[user_input]
    #     vid_clip_name = r"D:\loaded_clips\clip_{}.avi".format(user_input)
    #     saveVideoClip(vid_clip_name, clip_frames)
    #     user_input = input("input frame number")
    # c = plt.imshow(np.transpose(clip_frames[0], (1, 2, 0)))

    # plt.show()


    # labels = clip_8[1]
    # print(len(labels), len(labels))
    # np_labels = np.array(labels).T
    # for i in range(np_labels.shape[0]):
    #     print(f"frame {i}:  {np.argmax(np_labels[i])}. number of labels:{np.sum(np_labels[i])}")
    # for frame_num in range(len(labels[0])):
    # print(labels[:frame_num])
