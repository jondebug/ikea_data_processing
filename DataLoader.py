
from torch.utils.data import Dataset
from pathlib import Path
import os
from utils import getNumRecordings, getListFromFile
import json

class HololensStreamRecBase():

    """Face Landmarks dataset."""
    def __init__(self,  dataset_path, furniture_list: list, action_list_filename='action_list.txt',
                 train_filename='all_train_dir_list.txt', test_filename='all_train_dir_list.txt', transform=None,
                 gt_annotation_filename = 'db_gt_annotations.json'):
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

        #indexing_files:
        self.gt_annotation_filename = os.path.join(dataset_path, 'indexing_files', gt_annotation_filename)
        self.action_list_filename = os.path.join(dataset_path, 'indexing_files', action_list_filename)
        self.train_filename = os.path.join(dataset_path, 'indexing_files', train_filename)
        self.test_filename = os.path.join(dataset_path, 'indexing_files', test_filename)

        # load lists from files:
        self.action_list = self.get_list_from_file(self.action_list_filename)
        self.action_list.sort()
        if "NA" in self.action_list:
            self.action_list.remove("NA")

        self.action_list.insert(0, "NA")  #  0 label for unlabled frames
        self.num_classes = len(self.action_list)
        self.trainset_video_list = self.getListFromFile(self.train_filename)
        self.testset_video_list = self.getListFromFile(self.test_filename)
        self.all_video_list = self.testset_video_list + self.trainset_video_list

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    def __len__(self):
        return 10

    def get_annotated_videos_table(self, device='all'):
        """
        fetch the annotated videos table from the database
        :param : device: string ['all', 'dev1', 'dev2', 'dev3']
        :return: annotated videos table
        """
        if device == 'all':
            return_table = self.cursor_vid.execute('''SELECT * FROM videos WHERE annotated = 1''')
        else:
            return_table = self.cursor_vid.execute('''SELECT * FROM videos WHERE annotated = 1 AND camera = ?''',
                                                   (device,))
        return return_table

    def get_video_annotations_table(self, video_idx):
        """
        fetch the annotation table of a specific video
        :param :  video_idx: index of the desired video
        :return: video annotations table
        """
        return self.cursor_annotations.execute('''SELECT * FROM annotations WHERE video_id = ?''', (video_idx,))

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
    def __init__(self, dataset_path, db_filename='ikea_annotation_db_full',
                 action_list_filename='atomic_action_list.txt',
                 action_object_relation_filename='action_object_relation_list.txt',
                 train_filename='train_cross_env.txt',
                 test_filename='test_cross_env.txt', transform=None, set='test', camera='dev3', frame_skip=1,
                 frames_per_clip=64, resize=None, mode='vid', input_type='rgb'):
        super().__init__( dataset_path, furniture_list: list, action_list_filename='action_list.txt',
                 train_filename='all_train_dir_list.txt', test_filename='all_train_dir_list.txt', transform=None)
        self.mode = mode
        self.transform = transform
        self.set = set
        self.camera = camera
        self.frame_skip = frame_skip
        self.frames_per_clip = frames_per_clip
        self.resize = resize
        self.input_type = input_type
        if self.set == 'train':
            self.video_list = self.trainset_video_list
        elif self.set == 'test':
            self.video_list = self.testset_video_list
        else:
            raise ValueError("Invalid set name")

        self.video_set = self.get_video_frame_labels()
        self.clip_set, self.clip_label_count = self.get_clips()

    def get_video_frame_labels(self):
        # Extract the label data from the database
        # outputs a dataset structure of (video_path, multi-label per-frame, number of frames in the video)
        video_table = self.get_annotated_videos_table(device=self.camera)
        vid_list = []
        for row in video_table:
            n_frames = int(row["nframes"])
            video_path = row['video_path']
            if self.input_type == 'depth':
                video_path = video_path.replace('images', 'depth')
            video_name = os.path.join(video_path.split('/')[0], video_path.split('/')[1])
            if self.mode == 'vid':
                video_full_path = os.path.join(self.dataset_path, video_path, 'scan_video.avi')
            else:
                video_full_path = os.path.join(self.dataset_path, video_path)
            if not video_name in self.video_list:
                continue
            if n_frames < 66 * self.frame_skip:  # check video length
                continue
            if not os.path.exists(video_full_path):  # check if frame folder exists
                continue

            label = np.zeros((self.num_classes, n_frames), np.float32)  # allow multi-class representation
            label[0, :] = np.ones((1, n_frames), np.float32)  # initialize all frames as background|transition
            video_id = row['id']
            annotation_table = self.get_video_annotations_table(video_id)
            for ann_row in annotation_table:
                atomic_action_id = ann_row["atomic_action_id"]  # map the labels
                object_id = ann_row["object_id"]
                action_id = self.get_action_id(atomic_action_id, object_id)
                end_frame = ann_row['ending_frame'] if ann_row['ending_frame'] < n_frames else n_frames
                if action_id is not None:
                    label[:, ann_row['starting_frame']:end_frame] = 0  # remove the background label
                    label[action_id, ann_row['starting_frame']:end_frame] = 1

            vid_list.append(
                (video_full_path, label, n_frames))  # 0 = duration - irrelevant for initial tests, used for start
        return vid_list

    def get_clips(self):
        # extract equal length video clip segments from the full video dataset
        clip_dataset = []
        label_count = np.zeros(self.num_classes)
        for i, data in enumerate(self.video_set):
            n_frames = data[2]
            n_clips = int(n_frames / (self.frames_per_clip * self.frame_skip))
            remaining_frames = n_frames % (self.frames_per_clip * self.frame_skip)
            for j in range(0, n_clips):
                for k in range(0, self.frame_skip):
                    start = j * self.frames_per_clip * self.frame_skip + k
                    end = (j + 1) * self.frames_per_clip * self.frame_skip
                    label = data[1][:, start:end:self.frame_skip]
                    label_count = label_count + np.sum(label, axis=1)
                    frame_ind = np.arange(start, end, self.frame_skip).tolist()
                    clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, 0))
            if not remaining_frames == 0:
                frame_pad = self.frames_per_clip - remaining_frames
                start = n_clips * self.frames_per_clip * self.frame_skip + k
                end = start + remaining_frames
                label = data[1][:, start:end:self.frame_skip]
                label_count = label_count + np.sum(label, axis=1)
                label = data[1][:, start - frame_pad:end:self.frame_skip]
                frame_ind = np.arange(start - frame_pad, end, self.frame_skip).tolist()
                clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, frame_pad))
        return clip_dataset, label_count

    def load_rgb_frames(self, video_full_path, frame_ind):
        # load video file and extract the frames
        frames = []
        # Open the video file
        if self.mode == 'vid':
            cap = cv2.VideoCapture(video_full_path)
        for i in frame_ind:
            if self.mode == 'vid':  # load from video file
                cap.set(1, i)
                ret, img = cap.read()
            else:  # load from image folder
                if self.input_type == 'rgb':
                    img_filename = os.path.join(video_full_path, str(i).zfill(6) + '.jpg')
                else:
                    img_filename = os.path.join(video_full_path, str(i).zfill(6) + '.png')
                img = cv2.imread(img_filename)
                # img = cv2.imread(img_filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            try:
                # if self.input_type == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # else:
                #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            except:
                # debugging
                raise ValueError("error occured while loading frame {} from video {}".format(i, video_full_path))

            if self.resize is not None:
                w, h, c = img.shape
                if w < self.resize or h < self.resize:
                    d = self.resize - min(w, h)
                    sc = 1 + d / min(w, h)
                    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
                img = cv2.resize(img, dsize=(self.resize, self.resize))  # resizing the images
                img = (img / 255.) * 2 - 1
            frames.append(img)
        if self.mode == 'vid':
            cap.release()
        return np.asarray(frames, dtype=np.float32)

    def video_to_tensor(self, pic):
        """Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (T x H x W x C)
        to a torch.FloatTensor of shape (C x T x H x W)

        Args:
             pic (numpy.ndarray): Video to be converted to tensor.
        Returns:
             Tensor: Converted video.
        """
        return torch.tensor(pic.transpose([3, 0, 1, 2]), dtype=torch.float32)

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.clip_set)

    def __getitem__(self, index):
        # 'Generate one sample of data'
        video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]

        imgs = self.load_rgb_frames(video_full_path, frame_ind)
        imgs = self.transform(imgs)

        return self.video_to_tensor(imgs), torch.from_numpy(labels), vid_idx, frame_pad
