from pathlib import Path
from glob import glob
import numpy as np
import os
import csv
from shutil import copyfile
from project_hand_eye_to_pv import *
import cv2
import numpy as np
# import cv2

from hand_defs import HandJointIndex


def createRecordingDirList(path, recording_directories_idx_target_file):
    if "_recDir" in path[-8:]:
        with open(recording_directories_idx_target_file, "a") as all_rec_idx_file:
            all_rec_idx_file.write(path + "\n")
        return

    for sub_dir in glob(rf"{path}\*\\"):
        print(f"calling process_all_recordings_in_path for path: {sub_dir}, continuing search for recording dir")
        createRecordingDirList(sub_dir)


def get_list_from_file(self, filename):
    """
    retrieve a list of lines from a .txt file
    :param :
    :return: list of atomic actions
    """
    with open(filename) as f:
        line_list = f.read().splitlines()
    # line_list.sort()
    return line_list

def getNumRecordings(w_path):
    """
    recursively iterate over directory to get number of sub dirs
    """
    return len([f for f in w_path.iterdir() if f.is_dir() and "_recDir" in str(w_path)[-8:]])


def removeOriginalPvImages(w_path):
    orig_pv_path = w_path / "pv"
    assert (w_path / "norm" / "pv").exists()
    orig_pv_images = [f.path for f in os.scandir(orig_pv_path) if os.path.splitext(f)[-1] == '.png']
    for orig_pv_img in orig_pv_images:
        os.remove(orig_pv_img)




def load_head_hand_eye_data(csv_path):
    joint_count = HandJointIndex.Count.value

    data = np.loadtxt(csv_path, delimiter=',')

    n_frames = len(data)
    timestamps = np.zeros(n_frames)
    head_transs = np.zeros((n_frames, 3))

    left_hand_transs = np.zeros((n_frames, joint_count, 3))
    left_hand_transs_available = np.ones(n_frames, dtype=bool)
    right_hand_transs = np.zeros((n_frames, joint_count, 3))
    right_hand_transs_available = np.ones(n_frames, dtype=bool)

    # origin (vector, homog) + direction (vector, homog) + distance (scalar)
    gaze_data = np.zeros((n_frames, 9))
    gaze_available = np.ones(n_frames, dtype=bool)

    for i_frame, frame in enumerate(data):
        timestamps[i_frame] = frame[0]
        # head
        head_transs[i_frame, :] = np.array(frame[1:17].reshape((4, 4)))[:3, 3]
        # left hand
        left_hand_transs_available[i_frame] = (frame[17] == 1)
        left_start_id = 18
        for i_j in range(joint_count):
            j_start_id = left_start_id + 16 * i_j
            j_trans = np.array(frame[j_start_id:j_start_id + 16].reshape((4, 4))).T[:3, 3]
            left_hand_transs[i_frame, i_j, :] = j_trans

        # right hand
        right_hand_transs_available[i_frame] = (
            frame[left_start_id + joint_count * 4 * 4] == 1)
        right_start_id = left_start_id + joint_count * 4 * 4 + 1
        for i_j in range(joint_count):
            j_start_id = right_start_id + 16 * i_j
            j_trans = np.array(frame[j_start_id:j_start_id + 16].reshape((4, 4))).T[:3, 3]
            right_hand_transs[i_frame, i_j, :] = j_trans


        assert(j_start_id + 16 == 851)
        gaze_available[i_frame] = (frame[851] == 1)
        gaze_data[i_frame, :4] = frame[852:856]
        gaze_data[i_frame, 4:8] = frame[856:860]
        gaze_data[i_frame, 8] = frame[860]

    return (timestamps, head_transs, left_hand_transs, left_hand_transs_available,
            right_hand_transs, right_hand_transs_available, gaze_data, gaze_available)


def removeOriginalPlyFiles(w_path, sensor_name="Depth Long Throw"):

    orig_depth_path = w_path / "{}".format(sensor_name)

    assert (w_path / "norm" / "{}".format(sensor_name)).exists()
    orig_depth_ply_files = [f.path for f in os.scandir(orig_depth_path) if os.path.splitext(f)[-1] == '.ply']
    for orig_depth_ply in orig_depth_ply_files:
        os.remove(orig_depth_ply)
    #print(f"going to remove the following files:{orig_depth_ply_files}")

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
        # print(f"{frame_number}.png pv file exists")
        return
    copyfile(original_pv_path, norm_pv_path)


def copyRenameDepthImage(w_path, depth_timestamp, frame_number, sensor_name="Depth Long Throw"):

    for file_format in ["pgm", "ply"]:
        original_depth_path = Path(w_path / sensor_name / f"{depth_timestamp}.{file_format}")
        norm_depth_path = Path(w_path / "norm" / sensor_name / f"{frame_number}.{file_format}")
        if norm_depth_path.exists():
            # print(f"{frame_number}.{file_format} depth file exists")
            return
        copyfile(original_depth_path, norm_depth_path)


def copyRenameHandEyeImage(w_path, pv_to_depth_hand_eye_mapping):
    hand_eye_path = Path(w_path / "head_hand_eye.csv")
    norm_hand_eye_path = Path(w_path / "norm" / "head_hand_eye.csv")

    with open(hand_eye_path, 'r') as f, open(norm_hand_eye_path, 'w') as norm_f:
        csvreader = csv.reader(f)
        norm_csvreader = csv.writer(norm_f)
        hand_eye_dict = {}
        for row in csvreader:
            hand_eye_dict[int(row[0])] = row
        for frame_number, pv_timestamp in enumerate(pv_to_depth_hand_eye_mapping.keys()):
            depth_ts, hand_eye_ts = pv_to_depth_hand_eye_mapping[pv_timestamp]
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
    frame_timestamps = []
    for i_frame, frame in enumerate(lines[1:]):
        if 'nan' in frame:
            print(frame, "invalid pv header data")
            continue
        if len(frame) > 3:
            frame = frame.split(',')
            frame_timestamps.append(int(frame[0]))
    return frame_timestamps


def processHandEyeData(folder):
    print("processing hand eye")
    norm_proc_eye_path = Path(folder / "norm" / "norm_proc_eye_data.csv")
    norm_proc_hand_path = Path(folder / "norm" / "norm_proc_hand_data.csv")
    with open(norm_proc_eye_path, 'w') as norm_proc_eye_f, open(norm_proc_hand_path, 'w') as norm_proc_hand_f:
        eye_csvwriter = csv.writer(norm_proc_eye_f)
        hand_csvwriter = csv.writer(norm_proc_hand_f)

        head_hat_stream_path = list(folder.glob('*_eye.csv'))[0]
        print(f"opening eye data file {head_hat_stream_path}")

        pv_info_path = list(folder.glob('*pv.txt'))[0]
        pv_paths = sorted(list((folder / 'PV').glob('*png')))
        if len(pv_paths) == 0:
            print(f"this is an empty recording: {folder}")
            return -1

        # load head, hand, eye data
        (timestamps, _,
         left_hand_transs, left_hand_transs_available,
         right_hand_transs, right_hand_transs_available,
         gaze_data, gaze_available) = load_head_hand_eye_data(head_hat_stream_path)

        eye_str = " and eye gaze" if np.any(gaze_available) else ""

        # load pv info
        (frame_timestamps, focal_lengths, pv2world_transforms,
         ox, oy, width, height) = load_pv_data(pv_info_path)
        principal_point = np.array([ox, oy])
        n_frames = len(pv_paths)
        output_folder = folder / 'eye_hands'
        output_folder.mkdir(exist_ok=True)
        for pv_id in range(min(n_frames, len(focal_lengths))):
            eye_data_row = [pv_id]
            hand_data_row = [pv_id]
            pv_path = pv_paths[pv_id]
            sample_timestamp = int(str(pv_path.name).replace('.png', ''))

            hand_ts = match_timestamp(sample_timestamp, timestamps)
            # print('Frame-hand delta: {:.3f}ms'.format((sample_timestamp - timestamps[hand_ts]) * 1e-4))

            # img = cv2.imread(str(pv_path))
            # pinhole
            K = np.array([[focal_lengths[pv_id][0], 0, principal_point[0]],
                          [0, focal_lengths[pv_id][1], principal_point[1]],
                          [0, 0, 1]])
            try:
                Rt = np.linalg.inv(pv2world_transforms[pv_id])

            except np.linalg.LinAlgError:
                print('No pv2world transform')
                continue

            rvec, _ = cv2.Rodrigues(Rt[:3, :3])
            tvec = Rt[:3, 3]

            # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            hands = [(left_hand_transs, left_hand_transs_available),
                     (right_hand_transs, right_hand_transs_available)]
            for hand_id, hand in enumerate(hands):
                transs, avail = hand
                if avail[hand_ts]:
                    for joint_num, joint in enumerate(transs[hand_ts]):

                        hand_tr = joint.reshape((1, 3))
                        hand_data_row += list(hand_tr.reshape(3)) #adding 3d joint point to row.
                        # print(data_row)
                        xy, _ = cv2.projectPoints(hand_tr, rvec, tvec, K, None)
                        ixy = (int(xy[0][0][0]), int(xy[0][0][1]))
                        ixy = (width - ixy[0], ixy[1])
                        # print(ixy)
                        hand_data_row += [ixy[0], ixy[1]] #adding joint x,y projection to row.
                else:
                    hand_data_row += list(np.zeros(HandJointIndex.Count.value*5))

            if gaze_available[hand_ts]:
                point = get_eye_gaze_point(gaze_data[hand_ts])
                eye_data_row += list(gaze_data[hand_ts][:3]) #add origin_homog
                eye_data_row += list(point) #adding 3d pupil point to row.
                xy, _ = cv2.projectPoints(point.reshape((1, 3)), rvec, tvec, K, None)
                ixy = (int(xy[0][0][0]), int(xy[0][0][1]))
                ixy = (width - ixy[0], ixy[1])
                eye_data_row += [ixy[0], ixy[1]] #adding pupil x,y projection to row.
                if pv_id % 500 == 0:
                    print(width, ixy[0], ixy[1])
                    print(f"saving hand_eye processed data number {pv_id}")
            else:
                eye_data_row += [0, 0, 0, 0, 0, 0, 0, 0]

            eye_csvwriter.writerow(eye_data_row)
            hand_csvwriter.writerow(hand_data_row)
            #cv2.imwrite(str(output_folder / 'hands') + 'proj{}.png'.format(str(sample_timestamp).zfill(4)), img)
            # cv2.imwrite(f"{output_folder}/{str(sample_timestamp)}.png", img)







