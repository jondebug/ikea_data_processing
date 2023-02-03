from pathlib import Path
from utils import *
from glob import glob
import numpy as np
import os
import plyfile
import time
from multiprocessing import Pool, TimeoutError


def handleSinglePly(target_file, num_fps_points = 4096):



def createFpsRecCpy(long_throw_dir, target_dir, num_fps_points = 4096):

    ply_file_list = [os.path.join(long_throw_dir, long_throw_data)for long_throw_data in os.listdir(long_throw_dir)
                                if ".ply" in long_throw_data]
    # print(ply_file_list)
    if not os.path.exists(target_dir):
        print("making target dir: ", target_dir)
        os.makedirs(target_dir)
    #TODO: change this back to ply_file_list!!!!!!!!!
    for filenum, file in enumerate(ply_file_list):
        assert os.path.exists(file)
        # print(f'{ply_idx}.ply', file.split("\\")[-1])
        # assert f'{ply_idx}.ply' in file
        filename = file.split("\\")[-1]
        print(file, filename)
        plydata = plyfile.PlyData.read(file)
        # print(plydata)
        d = np.asarray(plydata['vertex'].data)
        pc = np.column_stack([d[p.name] for p in plydata['vertex'].properties])

        # print(pc)
        # start = time.process_time()
        #
        # fps_points = fps(n_points=num_fps_points, points=pc)
        # end_fps = time.process_time()
        # print("regular fps time: ", end_fps-start)

        end_fps = time.process_time()
        # batch = zip
        # with Pool(processes=16) as pool:
        #     for i in pool.imap_unordered(fps_ne, (pc, num_fps_points)):
        #         print(i)
        fps_points = fps_ne(npoint=num_fps_points, points=pc, stochastic_sample=False)


        end_ne_fps = time.process_time()
        print("ne fps time: ", end_ne_fps - end_fps)

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
        pts = list(zip(fps_points[:,0], fps_points[:,1], fps_points[:,2], fps_points[:,3],
                       fps_points[:,4], fps_points[:,5], fps_points[:,6], fps_points[:,7], fps_points[:,8]))
        # print(pts[:10])
        # print(pts[2])
        vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                      ('red', 'B'), ('green', 'B'), ('blue', 'B')])
        el = plyfile.PlyElement.describe(vertex, 'vertex')
        plyfile.PlyData([el], text="").write(os.path.join(target_dir,filename))




def createSmallDataset(src_dataset, target_dataset, furniture_modalities):


    print(src_dataset[-9:], target_dataset[-13:], '\HoloLens', target_dataset[-13:] == '\SmallDataset',
          src_dataset[-9:] == '\HoloLens', )
    assert target_dataset[-13:] == '\SmallDataset'
    assert src_dataset[-9:] == '\HoloLens'

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
            src_long_throw_dir = os.path.join(norm_furniture_rec_dir, "Depth Long Throw")
            target_long_throw_dir = os.path.join(target_furniture_dir, "Depth Long Throw")
            createFpsRecCpy(src_long_throw_dir, target_dir=target_long_throw_dir)
            return
        # print(furniture_rec_list)

if __name__ == '__main__':
    # w_path = Path(r'C:\HoloLens')
    furniture_modalities = ["Stool", "Drawer", "Table", "Coffee_Table"]
    src_dataset = r'D:\HoloLens'
    target_dataset = r'C:\SmallDataset'
    createSmallDataset(src_dataset, target_dataset, furniture_modalities)
