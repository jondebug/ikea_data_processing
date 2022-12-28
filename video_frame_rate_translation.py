import json
import os
from utils import getListFromFile

def createTranslationJson(all_timestamps, pv_timestamps, rec_dir):
    trans = {}
    video_frame_num = 0

    for pv_frame_num, pv_timestamp in enumerate(pv_timestamps):
        while video_frame_num < len(all_timestamps) and all_timestamps[video_frame_num] <= pv_timestamp:
            trans[video_frame_num] = pv_frame_num
            video_frame_num += 1

    while video_frame_num < len(all_timestamps):
        print("there are depth frames taken after last pv frame")
        trans[video_frame_num] = len(pv_timestamps)-1
        video_frame_num += 1
    # print(trans)
    with open(os.path.join(rec_dir, "frame_rate_translation.json"), "w") as new_json_file_obj:
        trans_json = json.dumps(trans)
        new_json_file_obj.write(trans_json)
        # return
    return trans

def createSingleFrameRateTranslation(rec_dir):
    all_timestamps = []
    pv_timestamps = []
    assert os.path.exists(os.path.join(rec_dir, 'pv'))
    for file in os.listdir(os.path.join(rec_dir, 'pv')):
        filename = os.path.splitext(os.path.basename(file))[0]
        all_timestamps.append(filename)
        pv_timestamps.append(filename)

    for file in os.listdir(os.path.join(rec_dir,  'Depth Long Throw')):
        if (file.endswith(".pgm")  and not file.endswith("_ab.pgm")):
            filename = os.path.splitext(os.path.basename(file))[0]
            all_timestamps.append(filename)
    all_timestamps.sort()
    # print(f"got {len(all_timestamps)} timestamps:\n all: {all_timestamps} \npv: {pv_timestamps}")
    return createTranslationJson(all_timestamps, pv_timestamps, rec_dir)



def createAllVideoFrameRateTranslation(dataset_top_dir):
    indexing_files_path = os.path.join(dataset_top_dir, "indexing_files")
    rec_dir_list_path = os.path.join(indexing_files_path, "all_recording_dir_list.txt")
    rec_dir_list = getListFromFile(rec_dir_list_path)
    for _dir_ in rec_dir_list:
        # if os.path.exists(os.path.join(_dir_, 'frame_rate_translation.json')):
        #     print(_dir_)
        trans = createSingleFrameRateTranslation(_dir_)





if __name__ == "__main__":
    dataset_dir = r'C:\Hololens'
    createAllVideoFrameRateTranslation(dataset_dir)
    # print(trans[138])
