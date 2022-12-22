from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import os
from utils import getNumRecordings, getListFromFile, getNumFrames
import json


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

        self.action_list_filename = os.path.join(dataset_path, 'indexing_files', action_list_filename)
        self.furniture_dir_sizes = [getNumRecordings(_dir_) for _dir_ in self.furniture_dirs]
        self.num_recordings = sum(self.furniture_dir_sizes)

        # indexing_files:
        self.gt_annotation_filename = os.path.join(dataset_path, 'indexing_files', gt_annotation_filename)
        self.action_list_filename = os.path.join(dataset_path, 'indexing_files', action_list_filename)
        self.train_filename = os.path.join(dataset_path, 'indexing_files', train_filename)
        self.test_filename = os.path.join(dataset_path, 'indexing_files', test_filename)

        # load lists from files:
        self.action_list = self.get_list_from_file(self.action_list_filename)
        self.action_list.sort()
        if "NA" in self.action_list:
            self.action_list.remove("NA")

        self.action_list.insert(0, "NA")  # 0 label for unlabled frames
        self.num_classes = len(self.action_list)
        self.train_video_list = self.getListFromFile(self.train_filename)
        self.test_video_list = self.getListFromFile(self.test_filename)
        self.all_video_list = self.testset_video_list + self.trainset_video_list
        self.action_id_mapping = {}
        for action, action_id in enumerate(self.action_list): self.action_id_mapping[action] = action_id
        print(self.action_id_mapping)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    def __len__(self):
        return 10

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
        with open(self.gt_annotation_filename) as json_file_obj:
            db_gt_annotations = json.load(json_file_obj)

        for _dir_ in rec_list:
            row = {"nframes": getNumFrames(_dir_), 'video_path': _dir_}

            if dataset != all:
                assert dataset in db_gt_annotations["database"][_dir_]["subset"]
            video_data_table.append(row)

    def get_video_annotations_table(self, video_path):
        with open(self.gt_annotation_filename) as json_file_obj:
            db_gt_annotations = json.load(json_file_obj)

        return db_gt_annotations["database"][video_path]["annotations"]

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
                 gt_annotation_filename='db_gt_annotations.json', modalities="all", frame_skip=1, frames_per_clip=32,
                 dataset="train"):

        super().__init__(dataset_path, furniture_list, action_list_filename,
                         train_filename, test_filename, transform, gt_annotation_filename)

        # self.camera = camera
        # self.resize = resize
        # self.input_type = input_type

        self.modalities = modalities
        self.transform = transform
        self.set = dataset
        self.frame_skip = frame_skip
        self.frames_per_clip = frames_per_clip

        if self.set == 'train':
            self.video_list = self.trainset_video_list
        elif self.set == 'test':
            self.video_list = self.testset_video_list
        else:
            raise ValueError("Invalid set name")

        self.video_set = self.get_video_frame_labels()
        # TODO: enable clip_set
        # self.clip_set, self.clip_label_count = self.get_clips()

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
            video_id = row['id']
            annotation_table = self.get_video_annotations_table(video_path)
            for ann_row in annotation_table:
                action = ann_row["label"]  # map the labels
                action_id = self.action_id_mapping[action]
                # object_id = ann_row["object_id"]
                # action_id = self.get_action_id(atomic_action_id, object_id)
                start_frame = ann_row['segment'][0]
                end_frame = ann_row['segment'][1]
                end_frame = end_frame if end_frame < n_frames else n_frames
                if action is not None:
                    rec_frame_labels[:, start_frame:end_frame] = 0  # remove the N/A
                    rec_frame_labels[action_id, ann_row['starting_frame']:end_frame] = 1

            vid_list.append(
                (video_path, rec_frame_labels, n_frames))  # 0 = duration - irrelevant for initial tests, used for start
        return vid_list
    #
    # def get_clips(self):
    #     # extract equal length video clip segments from the full video dataset
    #     clip_dataset = []
    #     label_count = np.zeros(self.num_classes)
    #     for i, data in enumerate(self.video_set):
    #         n_frames = data[2]
    #         n_clips = int(n_frames / (self.frames_per_clip * self.frame_skip))
    #         remaining_frames = n_frames % (self.frames_per_clip * self.frame_skip)
    #         for j in range(0, n_clips):
    #             for k in range(0, self.frame_skip):
    #                 start = j * self.frames_per_clip * self.frame_skip + k
    #                 end = (j + 1) * self.frames_per_clip * self.frame_skip
    #                 label = data[1][:, start:end:self.frame_skip]
    #                 label_count = label_count + np.sum(label, axis=1)
    #                 frame_ind = np.arange(start, end, self.frame_skip).tolist()
    #                 clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, 0))
    #         if not remaining_frames == 0:
    #             frame_pad = self.frames_per_clip - remaining_frames
    #             start = n_clips * self.frames_per_clip * self.frame_skip + k
    #             end = start + remaining_frames
    #             label = data[1][:, start:end:self.frame_skip]
    #             label_count = label_count + np.sum(label, axis=1)
    #             label = data[1][:, start - frame_pad:end:self.frame_skip]
    #             frame_ind = np.arange(start - frame_pad, end, self.frame_skip).tolist()
    #             clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, frame_pad))
    #     return clip_dataset, label_count
    #
    # def load_rgb_frames(self, video_full_path, frame_ind):
    #     # load video file and extract the frames
    #     frames = []
    #     # Open the video file
    #     if self.mode == 'vid':
    #         cap = cv2.VideoCapture(video_full_path)
    #     for i in frame_ind:
    #         if self.mode == 'vid':  # load from video file
    #             cap.set(1, i)
    #             ret, img = cap.read()
    #         else:  # load from image folder
    #             if self.input_type == 'rgb':
    #                 img_filename = os.path.join(video_full_path, str(i).zfill(6) + '.jpg')
    #             else:
    #                 img_filename = os.path.join(video_full_path, str(i).zfill(6) + '.png')
    #             img = cv2.imread(img_filename)
    #             # img = cv2.imread(img_filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    #         try:
    #             # if self.input_type == 'rgb':
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             # else:
    #             #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #         except:
    #             # debugging
    #             raise ValueError("error occured while loading frame {} from video {}".format(i, video_full_path))
    #
    #         if self.resize is not None:
    #             w, h, c = img.shape
    #             if w < self.resize or h < self.resize:
    #                 d = self.resize - min(w, h)
    #                 sc = 1 + d / min(w, h)
    #                 img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    #             img = cv2.resize(img, dsize=(self.resize, self.resize))  # resizing the images
    #             img = (img / 255.) * 2 - 1
    #         frames.append(img)
    #     if self.mode == 'vid':
    #         cap.release()
    #     return np.asarray(frames, dtype=np.float32)
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
    #
    # def __len__(self):
    #     # 'Denotes the total number of samples'
    #     return len(self.clip_set)
    #
    # def __getitem__(self, index):
    #     # 'Generate one sample of data'
    #     video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]
    #
    #     imgs = self.load_rgb_frames(video_full_path, frame_ind)
    #     imgs = self.transform(imgs)
    #
    #     return self.video_to_tensor(imgs), torch.from_numpy(labels), vid_idx, frame_pad
