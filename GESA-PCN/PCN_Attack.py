"""

Author: Most Husne Jahan
August -2023


Generate PCN-Attack Dataset:

We used only the complete point cloud from PCN dataset and apply Monotonic attack(packet loss) to create our own partial 
point cloud for training, validation and test set.

Ref: https://github.com/wentaoyuan/pcn/blob/master/io_util.py

The PCN Dataset is a widely used point cloud completion benchmark that contains 30974 models of 8 categories.

It contains 28974 training samples while each complete samples corresponds to 8 viewpoint partial scans, 800 validation samples and 1200 testing samples. Each ground-truth point cloud contains 16384 points, which are evenly sampled from
the shape surface.

We used only the complete point cloud from PCN dataset and apply Monotonic attack(packet loss) 
to create our own partial point cloud for training, validation set.

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

    V1= Packet loss= 26
"""
#===========================================================================
#===========================================================================

from Gilbert_Attack import *
import argparse
import os
import open3d as o3d
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--train_folder_source', type=str, default='GESA-PCA-PCN/train/GT', help='train source')
parser.add_argument('--train_folder_des', type=str, default='GESA-PCA-PCN/train/attack', help='train destination')

parser.add_argument('--val_folder_source', type=str, default='GESA-PCA-PCN/val/GT', help='val source')
parser.add_argument('--val_folder_des', type=str, default='GESA-PCA-PCN/val/attack', help='val destination')

parser.add_argument('--test_folder_source', type=str, default='GESA-PCA-PCN/test/GT', help='test source')
parser.add_argument('--test_folder_des', type=str, default='GESA-PCA-PCN/test/attack', help='test destination')

parser.add_argument('--mode', type=str, default='Train', help='Train or Val or Test')
args = parser.parse_args()


########## Create path #####################  

TRAIN_COMPLETE_POINTS_PATH = args.train_folder_source
TRAIN_OUT_PATH = args.train_folder_des

VAL_COMPLETE_POINTS_PATH = args.val_folder_source
VAL_OUT_PATH = args.val_folder_des

TEST_COMPLETE_POINTS_PATH = args.test_folder_source
TEST_OUT_PATH = args.test_folder_des


########## For PCN #####################  

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


############### Monotonic-Attack on Train Val Test ##################
def attack(COMPLETE_POINTS_PATH,OUT_PATH,Packet_Loss):

    print("Files and directories in a specified path:")

    if not os.path.exists(OUT_PATH):
        # os.mkdir(OUT_PATH)
        os.makedirs(OUT_PATH, exist_ok=True)

    if os.path.exists(OUT_PATH):

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
                    points, pcd= read_pcd(file_path)
                    # points =load_h5(file_path)
                    pointsfps = fps(points,2048)
                    # If Loss 50% then level is 50

                    level = Packet_Loss
                    print('Packet loss=',Packet_Loss)
                    print(f"\nApplying packet loss : {Packet_Loss}")

                    # Monotonic-Attack on Train Val Test
                    pointcloud = Monotonic_Attack(pointsfps,level)

                    # pointcloud = fps(pointcloud, 1024)

                    # print(pointcloud.shape)
                    #show_pcd(pointcloud) 
                    out_path = os.path.join(out_folder,file_name)
                    save_pcd("{}".format(out_path),pointcloud)
                    print("saved file {} \n".format(out_path))

    else:
         print("============ Attacked file Exists=============") 


############## Monotonic-Attack on Train Val #####################

if __name__ == '__main__':
    
    # Get Loss
    Packet_Loss_Total =  packet_loss()
    print('Packet Loss=',Packet_Loss_Total)
    
    Packet_Loss = round(Packet_Loss_Total[3])
    
    # Total_files =[]

    if args.mode=='Train':

        source_path = TRAIN_COMPLETE_POINTS_PATH
        out_folder = TRAIN_OUT_PATH+str(Packet_Loss)
        if not os.path.exists(out_folder):
           os.makedirs(out_folder, exist_ok=True)
    
        attack(source_path,out_folder,Packet_Loss)

    elif args.mode=='Val':    
        source_path = VAL_COMPLETE_POINTS_PATH
        out_folder = VAL_OUT_PATH+str(Packet_Loss)
        if not os.path.exists(out_folder):
           os.makedirs(out_folder, exist_ok=True)
        attack(source_path,out_folder,Packet_Loss)

    else:        
        source_path = TEST_COMPLETE_POINTS_PATH
        out_folder = TEST_OUT_PATH+str(Packet_Loss)
        if not os.path.exists(out_folder):
           os.makedirs(out_folder, exist_ok=True)
        attack(source_path,out_folder,Packet_Loss)    
    
    print('============= File conversion completed ==============')

########################## Monotonic-Attack on Train Val #####################################


