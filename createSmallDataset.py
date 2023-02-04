from pathlib import Path
from utils import *
from glob import glob
import numpy as np
import os
import plyfile
import time
import platform
from multiprocessing import Pool, TimeoutError
import sys
from joblib import Parallel, delayed
import multiprocessing



def cpyTxtFiles(src_norm_dir, target_dir):
    #copy json, eye_data, hand_data, combined_eye_hand data
    assert os.path.exists(src_norm_dir)
    w_src_orig_path = Path(src_norm_dir).parent
    print("searching path for json: ", w_src_orig_path)
    json_annotation_file = searchForAnnotationJson(str(w_src_orig_path))
    if not json_annotation_file:
        return
    annotation_file_name = json_annotation_file.split("\\")[-1]
    assert os.path.exists(json_annotation_file)
    print("found json annotation file: ", json_annotation_file, "named: ", annotation_file_name)
    src_file_list = [(json_annotation_file, annotation_file_name),
                     (os.path.join(src_norm_dir, "head_hand_eye.csv"),"head_hand_eye.csv"),
                     (os.path.join(src_norm_dir, "norm_proc_eye_data.csv"), "norm_proc_eye_data.csv"),
                     (os.path.join(src_norm_dir, "norm_proc_hand_data.csv"), "norm_proc_hand_data.csv")
                     ]
    if not os.path.exists(target_dir):
        print("making target dir: ", target_dir)
        os.makedirs(target_dir)

    for src_file, filename in src_file_list:
        target_file = os.path.join(target_dir,filename)
        print("creating file: ", target_file)
        copyfile(src_file, target_file)


def cpyPvFiles(src_dir, target_dir):

    src_pv_file_list = [pv_file for pv_file in os.listdir(src_dir)
                                if ".png" in pv_file]
    # print(ply_file_list)
    assert os.path.exists(src_dir)

    if not os.path.exists(target_dir):
        print("making target dir: ", target_dir)
        os.makedirs(target_dir)

    for src_file_name in src_pv_file_list:
        src_file_path = os.path.join(src_dir, src_file_name)
        target_file_path = os.path.join(target_dir, src_file_name)
        assert os.path.exists(src_file_path)
        assert os.path.exists(target_dir)
        copyfile(src_file_path, target_file_path)
    print("done creating pv folder for: ", target_dir)


def handleSinglePly(arg, num_fps_points=4096):
    print(arg)
    start = time.process_time()

    src_file, target_dir = arg
    assert os.path.exists(src_file)
    # print(f'{ply_idx}.ply', file.split("\\")[-1])
    # assert f'{ply_idx}.ply' in file
    print("platform: ", platform.platform())
    if "windows" in platform.platform():
        filename = src_file.split("\\")[-1]
    else:
        filename = src_file.split("/")[-1]
    print("crreating target file by joining: ", (target_dir, filename))
    target_file = os.path.join(target_dir, filename)
    print(target_file)
    if os.path.exists(target_file):
        print("file already exists: ", target_file)
        return 0
    print(src_file, filename)
    # handleSinglePly(src_file, os.path.join(target_dir, filename), num_fps_points=4096)

    plydata = plyfile.PlyData.read(src_file)
    # print(plydata)
    d = np.asarray(plydata['vertex'].data)
    pc = np.column_stack([d[p.name] for p in plydata['vertex'].properties])
    start_fps = time.process_time()

    fps_points = fps_ne(npoint=num_fps_points, points=pc, stochastic_sample=False)
    print("fps time:", time.process_time() - start_fps)

    pts = list(zip(fps_points[:, 0], fps_points[:, 1], fps_points[:, 2], fps_points[:, 3],
                   fps_points[:, 4], fps_points[:, 5], fps_points[:, 6], fps_points[:, 7], fps_points[:, 8]))
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                  ('red', 'B'), ('green', 'B'), ('blue', 'B')])
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text="").write(target_file)
    print("total process time:", time.process_time() - start)
    return(target_file, time.process_time() - start)

def createFpsRecCpy(long_throw_dir, target_dir, num_fps_points = 4096):

    ply_file_list = [os.path.join(long_throw_dir, long_throw_data)for long_throw_data in os.listdir(long_throw_dir)
                                if ".ply" in long_throw_data]
    # print(ply_file_list)
    if not os.path.exists(target_dir):
        print("making target dir: ", target_dir)
        os.makedirs(target_dir)
    #TODO: change this back to ply_file_list!!!!!!!!!
    start = time.process_time()

    #####
    # multiprocessing:
    #####
    # num_cores = multiprocessing.cpu_count()
    num_cores = 4
    print(f"machine has {num_cores}")
    arg_zip = list(zip(ply_file_list, [target_dir for _ in range (len(ply_file_list))]))
    Parallel(n_jobs=num_cores)(
        delayed(handleSinglePly)(arg_zip[i]) for i in range(len(ply_file_list)))

    return
    #multiprocessing.cpu_count()
        # Parallel(n_jobs=num_cores)(delayed(compute_point_clouds)(depth_ins_params, rgb_ins_params, depth_frames, rgb_frames, j, point_cloud_dir_name) for j, _ in enumerate(depth_frames))
    # with Pool(processes=3) as pool:
    #     for i in pool.imap_unordered(handleSinglePly, arg_zip):
    #         print(i)


    # for src_file in ply_file_list:
    #     print(handleSinglePly((src_file, target_dir)))

    # end_fps = time.process_time()
    # print("ne fps time: ", end_fps - start)
        # print(pc)
        #
        # fps_points = fps(n_points=num_fps_points, points=pc)
        # print("regular fps time: ", end_fps-start)

        # end_fps = time.process_time()
        # batch = zip
        #
        #     for i in pool.imap_unordered(fps_ne, (pc, num_fps_points)):
        #         print(i)


        # end_ne_fps = time.process_time()

        # fps_points = fps_ne(npoint=num_fps_points, points=pc, stochastic_sample=True, stochastic_sample_ratio_inv=4)
        # end_stochastic_fps = time.process_time()
        # print("stochastic ne fps time: ", end_stochastic_fps - end_ne_fps)


        # fps_np_points = fps_np(npoint=num_fps_points, points=pc)
        # end_np_fps = time.process_time()
        # print("np fps time: ", end_np_fps - end_ne_fps )
        #
        # print(fps_points.shape)
        # print(fps_points[0])
        # print(fps_points[:10])

        # print(pts[:10])
        # print(pts[2])




def createSmallDataset(src_dataset, target_dataset, furniture_modalities):


    print(src_dataset[-8:], target_dataset[-12:], 'HoloLens', target_dataset[-13:] == 'SmallDataset',
          src_dataset[-8:] == 'HoloLens', )
    assert target_dataset[-12:] == 'SmallDataset'
    assert src_dataset[-8:] == 'HoloLens'

    for furniture_name in furniture_modalities:
        furniture_src_dir = os.path.join(src_dataset, furniture_name)
        furniture_target_dir = os.path.join(target_dataset, furniture_name)
        norm_furniture_rec_list = [os.path.join(furniture_src_dir, _dir, "norm")for _dir in os.listdir(furniture_src_dir)
                             if "_recDir" in _dir]
        target_furniture_rec_list = [os.path.join(furniture_target_dir, _dir)for _dir in os.listdir(furniture_src_dir)
                             if "_recDir" in _dir]
        for target_furniture_dir, norm_furniture_rec_dir in zip(target_furniture_rec_list, norm_furniture_rec_list):
            if not os.path.exists(norm_furniture_rec_dir):
                print(norm_furniture_rec_dir)
                assert False
            src_norm_long_throw_dir = os.path.join(norm_furniture_rec_dir, "Depth Long Throw")
            target_long_throw_dir = os.path.join(target_furniture_dir, "Depth Long Throw")
            createFpsRecCpy(src_norm_long_throw_dir, target_dir=target_long_throw_dir)
            # src_norm_pv = os.path.join(norm_furniture_rec_dir, "pv")
            # target_pv = os.path.join(target_furniture_dir, "pv")
            # cpyPvFiles(src_norm_pv, target_pv)
            # cpyTxtFiles(norm_furniture_rec_dir, target_furniture_dir)
            # return
        # print(furniture_rec_list)

if __name__ == '__main__':
    # w_path = Path(r'C:\HoloLens')
    # print()
    # exit()
    # mod = sys.argv[1]
    mod = "Coffee_Table"
    furniture_modalities = ["Stool", "Drawer", "Table", "Coffee_Table"]
    assert mod in furniture_modalities

    if "windows" in platform.platform():
        src_dataset = r'D:\HoloLens'
        target_dataset = r'C:\SmallDataset'
    else:
        #linux:
        target_dataset = r'/mnt/c/SmallDataset'
        src_dataset= r'/mnt/d/HoloLens'
    createSmallDataset(src_dataset, target_dataset, [mod])
