"""

Author: Most Husne Jahan
August -2023


To downsample PCN Dataset:

python PCN_PCDDownAll.py

The PCN Dataset is a widely used point cloud completion benchmark that contains 30974 models of 8 categories.

It contains 28974 training samples while each complete samples corresponds to 8 viewpoint partial scans, 800 validation samples and 1200 testing samples. Each ground-truth point cloud contains 16384 points, which are evenly sampled from
the shape surface.

    category_ids = {
            "airplane"  : "02691156",  
            "cabinet"   : "02933112",  
            "car"       : "02958343",
            "chair"     : "03001627",
            "lamp"      : "03636649",
            "sofa"      : "04256520",
            "table"     : "04379243",
            "vessel"    : "04530566",  # boat
    }

"""
#===========================================================================
#===========================================================================

import argparse
import os
import open3d as o3d
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--train_complete_source', type=str, default='ShapeNetCompletion/train/complete', help='train complete source')
parser.add_argument('--train_complete_des', type=str, default='ShapeNetCompletion2048/train/complete', help='train complete destination')

parser.add_argument('--val_complete_source', type=str, default='ShapeNetCompletion/val/complete', help='val complete source')
parser.add_argument('--val_complete_des', type=str, default='ShapeNetCompletion2048/val/complete', help='val complete destination')

parser.add_argument('--test_complete_source', type=str, default='ShapeNetCompletion/test/complete', help='test complete source')
parser.add_argument('-test_complete_des', type=str, default='ShapeNetCompletion2048/test/complete', help='test complete destination')


parser.add_argument('--mode', type=str, default='Train', help='Train or Val or Test')
args = parser.parse_args()


########## Create path #####################  
train_complete_source = args.train_complete_source
train_complete_des = args.train_complete_des

val_complete_source = args.val_complete_source
val_complete_des = args.val_complete_des

test_complete_source = args.test_complete_source
test_complete_des = args.test_complete_des

########## For PCN Complete/GT Downsample #####################  

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    
    points = np.array(pcd.points)
    
    return points,pcd


def save_pcd(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd) 
    

def show_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

################ FPS #######################################################

def fps(ptcloud, n_points):
    
    """
    Input:
        xyz: pointcloud data, [N, D]
        n_points: number of samples
    Return:
        centroids: sampled pointcloud index, [n_points, D]
    """
    N, D = ptcloud.shape
    xyz = ptcloud[:, :3]
    centroids = np.zeros((n_points,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_points):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    ptcloud = ptcloud[centroids.astype(np.int32)]
    #print('Using FPS on the dataset')
    return ptcloud

############################ FPS #############################


def downsample(COMPLETE_POINTS_PATH,OUT_PATH):

    if not os.path.exists(OUT_PATH):
        # os.mkdir(OUT_PATH)
        os.makedirs(OUT_PATH, exist_ok=True)

    # if os.path.exists(OUT_PATH):


        for root, folders, files in os.walk(COMPLETE_POINTS_PATH):   
            for folder in folders:
                print(os.path.join(root, folder))
                full_path = os.path.join(OUT_PATH,folder)
                if not os.path.exists(full_path):
                    os.mkdir(full_path)

        for fname in os.listdir(COMPLETE_POINTS_PATH):

            # build the path to the folder
            folder_path = os.path.join(COMPLETE_POINTS_PATH, fname)
            out_folder = os.path.join(OUT_PATH,fname)

            if os.path.isdir(folder_path):
                # we are sure this is a folder; now lets iterate it
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    root, ext = os.path.splitext(file_path)
                    if ext in ['.pcd']:
                        points, pcd= read_pcd(file_path)
                        # points =load_h5(file_path)
                        pointcloud = fps(points,2048)
                        # pointcloud = fps(points, 1024)

                        print(pointcloud.shape)
                        #show_pcd(pointcloud) 
                        out_path = os.path.join(out_folder,file_name)
                        save_pcd("{}".format(out_path),pointcloud)
                        print("saved file {} \n".format(out_path))

    else:
         print("============ GT Downsamles file Exists=============") 


########## For PCN Complete/GT Downsample #####################  

if __name__ == '__main__':
    

    if args.mode=='Train':

        train_complete_source = args.train_complete_source
        train_complete_des = args.train_complete_des

        downsample(train_complete_source, train_complete_des)
       
    elif args.mode=='Val': 

        val_complete_source = args.val_complete_source
        val_complete_des = args.val_complete_des
     
        downsample(val_complete_source, val_complete_des)
        
    else: 

        test_complete_source = args.test_complete_source
        test_complete_des = args.test_complete_des
      
        downsample(test_complete_source, test_complete_des)
      
    print('============= File conversion completed ==============')

########################## Monotonic-Attack on Train Val #####################################


