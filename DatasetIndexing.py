import json
# from pathlib import Path
import os
from utils import createRecordingDirList

def createAllIndexingFiles(dataset_dir):
    # w_path = Path(dataset_dir)
    indexing_files_path = os.path.join(dataset_dir, "indexing_files")

    try:
        os.mkdir(indexing_files_path)
    except OSError as error:
        print(error)

    recording_dir_list_path = os.path.join(indexing_files_path, "recording_dir_list.txt")
    createRecordingDirList(path=indexing_files_path, recording_directories_idx_target_file=recording_dir_list_path)



if __name__ == "__main__":
    work_dir = r'C:\HoloLens'
    createAllIndexingFiles(work_dir)