"""

Author: Most Husne Jahan
August -2023

Generate Generate GESA-Shape Dataset:

* we get 41,952 models for training and 10,518 models for testing. It has 55 categories.

* We used only the GT point cloud from ShapeNet-55 dataset and apply Monotonic attack(packet loss) 
 to create our own attack point cloud for training, validation set.

* category_ids = {
          {"02691156": "airplane", "02747177": "trash bin", "02773838": "bag", 
          "02801938": "basket", "02808440": "bathtub", "02818832": "bed", 
          "02828884": "bench", "02843684": "birdhouse", "02871439": "bookshelf", 
          "02876657": "bottle", "02880940": "bowl", "02924116": "bus", "02933112": "cabinet",
          "02942699": "camera", "02946921": "can", "02954340": "cap", "02958343": "car", 
          "02992529": "cellphone", "03001627": "chair", "03046257": "clock", 
          "03085013": "keyboard", "03207941": "dishwasher", "03211117": "display",
          "03261776": "earphone", "03325088": "faucet", "03337140": "file cabinet",
          "03467517": "guitar", "03513137": "helmet", "03593526": "jar", "03624134": "knife", 
          "03636649": "lamp", "03642806": "laptop", "03691459": "loudspeaker",
          "03710193": "mailbox", "03759954": "microphone", "03761084": "microwaves", 
          "03790512": "motorbike", "03797390": "mug", "03928116": "piano", 
          "03938244": "pillow", "03948459": "pistol", "03991062": "flowerpot",
          "04004475": "printer", "04074963": "remote", "04090263": "rifle", 
          "04099429": "rocket", "04225987": "skateboard", "04256520": "sofa",
          "04330267": "stove", "04379243": "table", "04401088": "telephone", 
          "04460130": "tower", "04468005": "train", "04530566": "watercraft", 
          "04554684": "washer"}
    }

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

parser.add_argument('--train_folder_source', type=str, default='ShapeNet-55_V2/train/complete', help='train source')
parser.add_argument('--train_folder_des', type=str, default='ShapeNet-55_V2/train/attack', help='train destination')

parser.add_argument('--test_folder_source', type=str, default='ShapeNet-55_V2/test/complete', help='test source')
parser.add_argument('--test_folder_des', type=str, default='ShapeNet-55_V2/test/attack', help='test destination')

parser.add_argument('--mode', type=str, default='Train', help='Train or Val or Test')
args = parser.parse_args()


########## Create path #####################  

TRAIN_COMPLETE_POINTS_PATH = args.train_folder_source
TRAIN_OUT_PATH = args.train_folder_des

# VAL_COMPLETE_POINTS_PATH = args.val_folder_source
# VAL_OUT_PATH = args.val_folder_des

TEST_COMPLETE_POINTS_PATH = args.test_folder_source
TEST_OUT_PATH = args.test_folder_des

# # Get Loss
# Packet_Loss =  packet_loss()
# print('Packet Loss=',Packet_Loss)


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

########## For Completion3D #####################  

# def save_pcd(filename, points):

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # new_filename = Path(filename).stem + ".pcd"
#     (prefix, sep, suffix) = filename.rpartition('.')

#     new_filename = prefix + '.pcd'
#     o3d.io.write_point_cloud(new_filename, pcd) 

# def load_h5(path, verbose=False):
#     if verbose:
#         print("Loading %s \n" % (path))
#     f = h5py.File(path, 'r')
#     cloud_data = np.array(f['data'])
#     f.close()

#     return cloud_data

##########  For Completion3D  #####################  

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
                    # os.mkdir(full_path)
                    os.makedirs(full_path, exist_ok=True)

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
                    print('Packet Loss=',Packet_Loss[0])
                    level= round(Packet_Loss[0])

                    print(f"Applying packet loss on file #: {file_path}")

                    # Monotonic-Attack on Train Val Test
                    pointcloud = Monotonic_Attack(pointsfps,level)

                    # pointcloud = fps(pointcloud, 1024)

                    print(pointcloud.shape)
                    #show_pcd(pointcloud) 
                    out_path = os.path.join(out_folder,file_name)
                    save_pcd("{}".format(out_path),pointcloud)
                    print("saved file {} \n".format(out_path))

    else:
         print("============ Attacked file Exists=============") 


############## Monotonic-Attack on Train Val #####################

if __name__ == '__main__':
    
    # Get Loss
    Packet_Loss =  packet_loss()
    print('Packet Loss=',Packet_Loss)


    if args.mode=='Train':

        source_path = TRAIN_COMPLETE_POINTS_PATH
        out_folder = TRAIN_OUT_PATH

        # train_folder_des = 'ShapeNet-55_processed/Train'

        # if not os.path.exists(TRAIN_OUT_PATH):
            # os.makedirs(TRAIN_OUT_PATH, exist_ok=True)

        attack(TRAIN_COMPLETE_POINTS_PATH,TRAIN_OUT_PATH,Packet_Loss)

    # elif args.mode=='Val':    
    #     source_path = VAL_COMPLETE_POINTS_PATH
    #     out_folder = VAL_OUT_PATH
    #     attack(source_path,out_folder,Packet_Loss)

    else:        
        source_path = TEST_COMPLETE_POINTS_PATH
        out_folder = TEST_OUT_PATH
        
        # if not os.path.exists(TEST_OUT_PATH):
            #   os.makedirs(TEST_OUT_PATH, exist_ok=True)

        attack(TEST_COMPLETE_POINTS_PATH,TEST_OUT_PATH,Packet_Loss)    
    
    print('============= File conversion completed ==============')

########################## Monotonic-Attack on Train Val #####################################

