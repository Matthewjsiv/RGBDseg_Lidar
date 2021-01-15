#DATASET: https://unmannedlab.github.io/research/RELLIS-3D
#download the full images, annotations, and split files
#create an empty dataset folder (might also need to create train, test, and validation subfolders)


import shutil
import os
import numpy as np
from pathlib import Path
from transformtest import load_from_bin, print_projection_plt, depth_color, points_filter, get_cam_mtx, get_lidar2cam_mtx, get_depth, get_im
cwd = os.getcwd()


f = open("train.lst","r")
lines = f.readlines()
num = 0
for line in lines:
    input,target = line.split(' ')
    target = target.replace('\n','')
    series = input.split('/')[0]
    id = input[input.find('frame') + 5: input.find('-')]
    #print(id)
    binname = 'Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/' + str(series) + '/os1_cloud_node_kitti_bin/' + str(id) + '.bin'
    res = get_depth('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, binname, series)
    dirname = 'ndataset/train/' + str(series) + '-' + str(id)
    #print(dirname)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    shutil.copy('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, dirname + '/in.jpg')
    shutil.copy('Rellis_3D_pylon_camera_node_label_id/Rellis-3D/' + target, dirname + '/target.png')
    np.save(dirname + '/depth.npy', res)
f.close()

f = open("test.lst","r")
lines = f.readlines()
for line in lines:
    input,target = line.split(' ')
    target = target.replace('\n','')
    series = input.split('/')[0]
    id = input[input.find('frame') + 5: input.find('-')]
    #print(id)
    binname = 'Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/' + str(series) + '/os1_cloud_node_kitti_bin/' + str(id) + '.bin'
    res = get_depth('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, binname, series)
    dirname = 'ndataset/test/' + str(series) + '-' + str(id)
    #print(dirname)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    shutil.copy('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, dirname + '/in.jpg')
    shutil.copy('Rellis_3D_pylon_camera_node_label_id/Rellis-3D/' + target, dirname + '/target.png')
    np.save(dirname + '/depth.npy', res)
f.close()

f = open("val.lst","r")
lines = f.readlines()
for line in lines:
    input,target = line.split(' ')
    target = target.replace('\n','')
    series = input.split('/')[0]
    id = input[input.find('frame') + 5: input.find('-')]
    #print(id)
    binname = 'Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/' + str(series) + '/os1_cloud_node_kitti_bin/' + str(id) + '.bin'
    res = get_depth('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, binname, series)
    dirname = 'ndataset/val/' + str(series) + '-' + str(id)
    #print(dirname)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    shutil.copy('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, dirname + '/in.jpg')
    shutil.copy('Rellis_3D_pylon_camera_node_label_id/Rellis-3D/' + target, dirname + '/target.png')
    np.save(dirname + '/depth.npy', res)
f.close()
