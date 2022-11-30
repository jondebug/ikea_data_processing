from pathlib import Path
from glob import glob
import numpy as np
import os
import csv
from shutil import copyfile

def build_normalized_data_dir(w_path, sensor_name="Depth Long Throw"):
    norm_dir = Path(w_path / "norm")
    norm_pv_dir = Path(w_path / "norm" / "pv")
    norm_depth_dir = Path(w_path / "norm" / sensor_name)

    if not norm_dir.exists():
        os.mkdir(norm_dir)

    if not norm_pv_dir.exists():
        os.mkdir(norm_pv_dir)

    if not norm_depth_dir.exists():
        os.mkdir(norm_depth_dir)


def copyRenamePvImage(w_path, pv_timestamp, frame_number):
    original_pv_path = Path(w_path / "pv" / f"{pv_timestamp}.png")
    norm_pv_path = Path(w_path / "norm" / "pv" / f"{frame_number}.png")
    if norm_pv_path.exists():
        print(f"{frame_number}.png pv file exists")
        return
    copyfile(original_pv_path, norm_pv_path)


def copyRenameDepthImage(w_path, depth_timestamp, frame_number, sensor_name="Depth Long Throw"):

    for file_format in ["pgm", "ply"]:
        original_depth_path = Path(w_path / sensor_name / f"{depth_timestamp}.{file_format}")
        norm_depth_path = Path(w_path / "norm" / sensor_name / f"{frame_number}.{file_format}")
        if norm_depth_path.exists():
            print(f"{frame_number}.{file_format} depth file exists")
            return
        copyfile(original_depth_path, norm_depth_path)


def copyRenameHandEyeImage(w_path, pv_to_depth_hand_eye_mapping):
    hand_eye_path = Path(w_path / "head_hand_eye.csv")
    norm_hand_eye_path = Path(w_path / "norm" / "head_hand_eye.csv")
    # print(f"opening pv file {hand_eye_path}")
    #hand_eye_timestamps = []

    with open(hand_eye_path, 'r') as f, open(norm_hand_eye_path, 'w') as norm_f:
        csvreader = csv.reader(f)
        norm_csvreader = csv.writer(norm_f)
        hand_eye_dict = {}
        for row in csvreader:
            hand_eye_dict[int(row[0])] = row
        for frame_number, pv_timestamp in enumerate(pv_to_depth_hand_eye_mapping.keys()):
            depth_ts, hand_eye_ts = pv_to_depth_hand_eye_mapping[pv_timestamp]
            # print("pv to hand eye ts: ",frame_number, ":", pv_timestamp, hand_eye_ts)
            norm_csvreader.writerow([frame_number] + hand_eye_dict[hand_eye_ts][1:])

#


def matchTimestamp(target, all_timestamps):
    return all_timestamps[np.argmin([abs(x - target) for x in all_timestamps])]


def extract_timestamp(path, depth_path_suffix):
    path = path.name.replace(depth_path_suffix, '')
    return int(path.split('.')[0])


def getHandEyeTimestamps(w_path):
    hand_eye_path = Path(w_path / "head_hand_eye.csv")
    # print(f"opening pv file {hand_eye_path}")
    hand_eye_timestamps = []
    with open(hand_eye_path, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            hand_eye_timestamps.append(int(row[0]))
    return hand_eye_timestamps


def getDepthTimestamps(w_path, sensor_name, depth_path_suffix):
    depth_path = Path(w_path / sensor_name)
    depth_paths = sorted(depth_path.glob('*[0-9]{}.pgm'.format(depth_path_suffix)))
    n_depth_frames = len(depth_paths)
    depth_timestamps = np.zeros(n_depth_frames, dtype=np.longlong)
    for i_path, path in enumerate(depth_paths):
        depth_timestamp = extract_timestamp(path, depth_path_suffix)
        depth_timestamps[i_path] = depth_timestamp
    return depth_timestamps

def getPvTimestamps(w_path):
    pv_csv_path = list(w_path.glob('*pv.txt'))[0]
    # print(f"opening pv file {pv_csv_path}")
    with open(pv_csv_path) as f:
        lines = f.readlines()
    if len(lines) <= 0:
        print(f"fount empty pv header file in: {pv_csv_path}")
        return
    n_frames = len(lines) - 1
    frame_timestamps = np.zeros(n_frames, dtype=np.longlong)
    for i_frame, frame in enumerate(lines[1:]):
        if 'nan' in frame:
            print(frame, "invalid pv header data")
            continue
        if len(frame) > 3:
            frame = frame.split(',')
            frame_timestamps[i_frame] = int(frame[0])
    return frame_timestamps







