from pathlib import Path
# from project_hand_eye_to_pv import project_hand_eye_to_pv
# from utils import check_framerates, extract_tar_file
# from save_pclouds import save_pclouds
# from convert_images import convert_images
from utils import *
from glob import glob
import numpy as np
import os


def createNormalizedFiles(rec_dir, pv_to_depth_hand_eye_mapping: dict, sensor_name="Depth Long Throw"):
    w_path = Path(rec_dir)
    pv_dir = w_path / "PV"
    depth_dir = w_path / "{}".format(sensor_name)
    assert pv_dir.exists() and depth_dir.exists() and (w_path / "head_hand_eye.csv").exists()
    build_normalized_data_dir(w_path)
    for frame_number, pv_timestamp in enumerate(pv_to_depth_hand_eye_mapping.keys()):
        depth_ts, hand_eye_ts = pv_to_depth_hand_eye_mapping[pv_timestamp]
        # print(depth_ts, hand_eye_ts)
        copyRenamePvImage(w_path, pv_timestamp, frame_number)
        copyRenameDepthImage(w_path, depth_ts, frame_number)

    copyRenameHandEyeImage(w_path, pv_to_depth_hand_eye_mapping)



def createPVtoDepthHandEyeMapping(rec_dir, depth_path_suffix='', sensor_name="Depth Long Throw"):
    # sub_dir_lst = glob(rf"{rec_dir}\*\\")
    w_path = Path(rec_dir)
    pv_dir = w_path / "PV"
    depth_dir = w_path / "{}".format(sensor_name)
    assert pv_dir.exists() and depth_dir.exists() and (w_path / "head_hand_eye.csv").exists()

    pv_timestamps = getPvTimestamps(w_path)
    depth_timestamps = getDepthTimestamps(w_path, sensor_name, depth_path_suffix)
    hand_eye_timestamps = getHandEyeTimestamps(w_path)
    # print(f'found the following {len(pv_timestamps)} PV timestamps: {pv_timestamps}')
    # print(f'found the following {len(depth_timestamps)} depth timestamps: {depth_timestamps}')

    pv_to_depth_hand_eye_mapping = {}
    for frame_number, pv_timestamp in enumerate(pv_timestamps):
        matching_depth_ts = matchTimestamp(target=pv_timestamp, all_timestamps=depth_timestamps)
        matching_hand_eye_ts = matchTimestamp(target=pv_timestamp, all_timestamps=hand_eye_timestamps)
        pv_to_depth_hand_eye_mapping[pv_timestamp] = (matching_depth_ts, matching_hand_eye_ts)
    return pv_to_depth_hand_eye_mapping


def normalizeAllRecordingsInPath(path):
    if "_recDir" in path[-8:]:
        pv_to_depth_hand_eye_mapping = createPVtoDepthHandEyeMapping(path)
        print("got mapping. creating normalized data dir for recording")
        createNormalizedFiles(rec_dir=path, pv_to_depth_hand_eye_mapping=pv_to_depth_hand_eye_mapping)
        print("normalized recording data done.")
        return

    for sub_dir in glob(rf"{path}\*\\"):
        print(f"calling process_all_recordings_in_path for path: {sub_dir}, continuing search for recording dir")
        normalizeAllRecordingsInPath(sub_dir)


if __name__ == '__main__':
    # w_path = Path(r'C:\HoloLens')
    normalizeAllRecordingsInPath(r'C:\HoloLens\Table')
