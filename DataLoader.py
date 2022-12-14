
from torch.utils.data import Dataset

class HololensStreamRecDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,  root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.root_dir = root_dir
        return

    def __len__(self):
        return 10



    def __getitem__(self, idx):
