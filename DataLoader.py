
from torch.utils.data import Dataset
from pathlib import Path
from utils import getNumRecordings
class HololensStreamRecDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,  root_dir, furniture_list: list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.root_dir = Path(root_dir)
        self.furniture_dirs = [(Path(root_dir) / furniture_name) for furniture_name in furniture_list]
        assert(Path(furniture_dir).exists() for furniture_dir in self.furniture_dirs)

        self.furniture_dir_sizes = [getNumRecordings(_dir) for _dir in self.furniture_dirs]
        self.num_recordings = len(self.furniture_dir_sizes)


        return

    def __len__(self):
        return 10


    #TODO: implement:
    # def __getitem__(self, idx):
    #
