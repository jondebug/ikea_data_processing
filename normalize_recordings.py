
from pathlib import Path
# from project_hand_eye_to_pv import project_hand_eye_to_pv
# from utils import check_framerates, extract_tar_file
# from save_pclouds import save_pclouds
# from convert_images import convert_images
from utils import *
from glob import glob
import os


def normalizeAllRecordingsInPath(path):

    if "_recDir" in path[-8:]:
        pv_to_depth_mapping = createPVtoDepthMapping(path)

        return

    for sub_dir in glob(rf"{path}\*\\"):
        print(f"calling process_all_recordings_in_path for path: {sub_dir}, continuing search for recording dir")
        normalizeAllRecordingsInPath(sub_dir)


if __name__ == '__main__':
    w_path = Path(r'C:\HoloLens')
    normalizeAllRecordingsInPath(r'C:\HoloLens')