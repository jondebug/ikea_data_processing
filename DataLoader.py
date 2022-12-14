
from torch.utils.data import Dataset
from pathlib import Path
import os
from utils import getNumRecordings, get_list_from_file


class HololensStreamRecDataset(Dataset):

    """Face Landmarks dataset."""
    def __init__(self,  dataset_path, furniture_list: list, action_list_filename='atomic_action_list.txt',
                 train_filename='train_cross_env.txt', test_filename='test_cross_env.txt', transform=None):
        """
        Args:
            action_list_filename (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.dataset_root_path = Path(dataset_path)
        self.furniture_dirs = [(Path(dataset_path) / furniture_name) for furniture_name in furniture_list]
        assert(Path(furniture_dir).exists() for furniture_dir in self.furniture_dirs)
        self.action_list_filename = os.path.join(dataset_path, 'indexing_files', action_list_filename)

        self.furniture_dir_sizes = [getNumRecordings(_dir) for _dir in self.furniture_dirs]
        self.num_recordings = len(self.furniture_dir_sizes)

        #indexing_files:
        self.action_list_filename = os.path.join(dataset_path, 'indexing_files', action_list_filename)
        self.train_filename = os.path.join(dataset_path, 'indexing_files', train_filename)
        self.test_filename = os.path.join(dataset_path, 'indexing_files', test_filename)

        # load lists from files:
        self.action_list = self.get_list_from_file(self.action_list_filename)
        self.action_list.sort()
        self.action_list.insert(0, "NA")  #  0 label for unlabled frames
        self.num_classes = len(self.action_list)
        self.trainset_video_list = self.get_list_from_file(self.train_filename)
        self.testset_video_list = self.get_list_from_file(self.test_filename)
        self.all_video_list = self.testset_video_list + self.trainset_video_list



    def __len__(self):
        return 10


    #TODO: implement:
    # def __getitem__(self, idx):
    #
