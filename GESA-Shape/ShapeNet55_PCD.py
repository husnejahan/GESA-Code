"""

Author: Most Husne Jahan
August -2023

Convert ShapeNet-55 Dataset to .PCD

"""

import logging

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import argparse
import open3d as o3d
import pandas as pd
from pathlib import Path
import numpy as np
import torch.utils.data as data
from ioPC import IO


class ShapeNet(data.Dataset):
    def __init__(self, root, out_folder, DATA_PATH='ShapeNet-55', PC_PATH='ShapeNet-55/shapenet_pc', npoints=2048, subset='train', save_file=True):
        self.data_root = os.path.join(root, DATA_PATH)
        self.pc_path = PC_PATH
        self.npoints = npoints
        self.subset = subset
        self.save_file = save_file
        self.out_folder = out_folder
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        idx = 0 
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]

            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
            self.getfile(idx)
            idx += 1

        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def save_pcd(self, out_path, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        name, file_extension = os.path.splitext(out_path)
        o3d.io.write_point_cloud(f'{name}.pcd', pcd)
        print(f"saved file {name}")

    def pc_norm(self, pc):
        """ Normalize point cloud to be centered at origin and scaled to unit sphere """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def getfile(self, idx):
        sample = self.file_list[idx]
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        data = self.pc_norm(data)
        point_set = data
        data = torch.from_numpy(data).float()

        label = sample['taxonomy_id']
        print(f'label={label}')
        name = sample['model_id']

        if self.save_file:
            full_path = os.path.join(self.out_folder, label)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            out_path = os.path.join(full_path, name)
            self.save_pcd(out_path, point_set)

        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder_des', type=str, default='ShapeNet-55_V2/train/complete', help='Train destination')
    parser.add_argument('--test_folder_des', type=str, default='ShapeNet-55_V2/test/complete', help='Test destination')
    args = parser.parse_args()

    root = os.getcwd()
    train_folder_des = args.train_folder_des
    test_folder_des = args.test_folder_des

    if not os.path.exists(train_folder_des):
        os.makedirs(train_folder_des)
    
    if not os.path.exists(test_folder_des):
        os.makedirs(test_folder_des)

    # Process training data
    d1 = ShapeNet(root=root, out_folder=train_folder_des, DATA_PATH='ShapeNet-55', PC_PATH='ShapeNet-55/shapenet_pc', npoints=2048, subset='train', save_file=True)
    filelist1 = d1.__len__()

    # Process test data
    d2 = ShapeNet(root=root, out_folder=test_folder_des, DATA_PATH='ShapeNet-55', PC_PATH='ShapeNet-55/shapenet_pc', npoints=2048, subset='test', save_file=True)
    filelist2 = d2.__len__()

    print(f"Train datasize: {filelist1}")
    print(f"Test datasize: {filelist2}")
    print('=================== All file conversion Done =====================')
