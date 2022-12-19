import json
# from pathlib import Path
import os
from utils import createAllRecordingDirList, createTrainTestFiles


def createSeperateFurnitureRecLists(dataset_dir):
    indexing_files_path = os.path.join(dataset_dir, "indexing_files")

    [createAllRecordingDirList(dataset_dir=os.path.join(dataset_dir, furniture_name),
                               target_file=os.path.join(dataset_dir, "{}_recording_dir_list.txt".format(furniture_name)))
     for furniture_name in os.listdir(dataset_dir)
     if os.path.isdir(os.path.join(dataset_dir, furniture_name)) and furniture_name != "indexing_files"]


def createAllIndexingFiles(dataset_dir):
    # w_path = Path(dataset_dir)
    indexing_files_path = os.path.join(dataset_dir, "indexing_files")

    try:
        os.mkdir(indexing_files_path)
    except OSError as error:
        print(error)

    recording_dir_list_path = os.path.join(indexing_files_path, "all_recording_dir_list.txt")
    createAllRecordingDirList(dataset_dir=indexing_files_path, target_file=recording_dir_list_path)
    createSeperateFurnitureRecLists(dataset_dir)
    createTrainTestFiles(dataset_dir=indexing_files_path)


if __name__ == "__main__":
    work_dir = r'C:\HoloLens'
    createAllIndexingFiles(work_dir)