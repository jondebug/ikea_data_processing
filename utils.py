from pathlib import Path
from glob import glob
import numpy as np
import os


#                 target_id = match_timestamp(timestamp, pv_timestamps)
#                 pv_ts = pv_timestamps[target_id]
#                 rgb_path = str(folder / 'PV' / f'{pv_ts}.png')
#                 assert Path(rgb_path).exists()
#                 pv_img = cv2.imread(rgb_path)

def match_timestamp(target, all_timestamps):
    return np.argmin([abs(x - target) for x in all_timestamps])

def createPVtoDepthMapping(rec_dir):

    sub_dir_lst = glob(rf"{rec_dir}\*\\")
    w_path = Path(rec_dir)
    sensor_name = "Depth Long Throw"
    pv_dir = w_path / "PV"
    depth_dir = w_path / "{}".format(sensor_name)
    assert pv_dir.exists() and depth_dir.exists()
            #and (w_path / "head_hand_eye.csv").exists()


    for sub_dir in sub_dir_lst:
        print(f"calling process_all_recordings_in_path for path: {sub_dir}, continuing search for recording dir")
        normalizeAllRecordingsInPath(sub_dir)
