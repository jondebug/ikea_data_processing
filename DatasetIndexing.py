import json
# from pathlib import Path
import os
from utils import createAllRecordingDirList, createTrainTestFiles, getListFromFile, writeListToFile, getAllJsonAnnotations


def createAnnotationJson(dataset_dir):
    all_annotations = getAllJsonAnnotations(dataset_dir=dataset_dir, merged_json={})
    print(all_annotations)



def copyActionList(dataset_dir, action_list_txt_file=""):

    if action_list_txt_file == "":
        action_list_txt_file = os.path.join(os.getcwd(), "action_list.txt")
    action_list = getListFromFile(action_list_txt_file)
    print(action_list)
    writeListToFile(filename=os.path.join(dataset_dir,"indexing_files", "action_list.txt"), line_list=action_list)


def createSeperateFurnitureRecLists(dataset_dir):
    indexing_files_path = os.path.join(dataset_dir, "indexing_files")

    [createAllRecordingDirList(dataset_dir=os.path.join(dataset_dir, furniture_name),
                               target_file=os.path.join(indexing_files_path, "{}_recording_dir_list.txt".format(furniture_name)))
     for furniture_name in os.listdir(dataset_dir)
     if os.path.isdir(os.path.join(dataset_dir, furniture_name)) and furniture_name != "indexing_files"]


def createAllIndexingFiles(dataset_dir):
    # w_path = Path(dataset_dir)
    # indexing_files_path = os.path.join(dataset_dir, "indexing_files")
    #
    # if not os.path.exists(indexing_files_path): os.mkdir(indexing_files_path)
    #
    # recording_dir_list_path = os.path.join(indexing_files_path, "all_recording_dir_list.txt")
    # createAllRecordingDirList(dataset_dir=dataset_dir, target_file=recording_dir_list_path)
    # createSeperateFurnitureRecLists(dataset_dir)
    # createTrainTestFiles(dataset_dir=dataset_dir)
    copyActionList(dataset_dir=dataset_dir)

if __name__ == "__main__":
    work_dir = r'C:\HoloLens'
    createAllIndexingFiles(work_dir)
