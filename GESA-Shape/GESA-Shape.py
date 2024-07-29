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

from GESA import PointCloudAttack  # Assuming PointCloudAttack is in GESA.py
import argparse
import os
import open3d as o3d
import numpy as np
from pathlib import Path

class GESA_ShapeDataGenerator:
    def __init__(self, packet_loss_level, attack_type='monotonic'):
        self.attack_simulator = PointCloudAttack(packet_loss_level)
        self.attack_type = attack_type

    def read_pcd(self, filename):
        pcd = o3d.io.read_point_cloud(filename)
        points = np.array(pcd.points)
        return points, pcd

    def save_pcd(self, filename, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd)

    def show_pcd(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

    def apply_attack(self, pointcloud):
        if self.attack_type == 'monotonic':
            level = round(self.attack_simulator.packet_loss_level / 10)
            return self.attack_simulator.monotonic_attack(pointcloud)
        elif self.attack_type == 'non_monotonic':
            return self.attack_simulator.non_monotonic_attack(pointcloud)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def process_directory(self, source_path, dest_path):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)

        for root, folders, files in os.walk(source_path):
            for folder in folders:
                full_path = os.path.join(dest_path, folder)
                if not os.path.exists(full_path):
                    os.makedirs(full_path, exist_ok=True)

        for fname in os.listdir(source_path):
            folder_path = os.path.join(source_path, fname)
            out_folder = os.path.join(dest_path, fname)

            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    points, _ = self.read_pcd(file_path)
                    points_fps = self.attack_simulator.fps(points, 2048)
                    attacked_points = self.apply_attack(points_fps)
                    out_path = os.path.join(out_folder, file_name)
                    self.save_pcd(out_path, attacked_points)
                    print(f"Saved file {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder_source', type=str, default='ShapeNet-55_V2/train/complete', help='Train source')
    parser.add_argument('--train_folder_des', type=str, default='ShapeNet-55_V2/train/attack', help='Train destination')
    parser.add_argument('--test_folder_source', type=str, default='ShapeNet-55_V2/test/complete', help='Test source')
    parser.add_argument('--test_folder_des', type=str, default='ShapeNet-55_V2/test/attack', help='Test destination')
    parser.add_argument('--packet_loss_level', type=int, required=True, help='Packet loss level')
    parser.add_argument('--mode', type=str, default='Train', choices=['Train', 'Test'], help='Mode: Train or Test')
    parser.add_argument('--attack_type', type=str, default='monotonic', choices=['monotonic', 'non_monotonic'], help='Type of attack')

    args = parser.parse_args()

    generator = GESA_ShapeDataGenerator(args.packet_loss_level, args.attack_type)

    if args.mode == 'Train':
        generator.process_directory(args.train_folder_source, args.train_folder_des)
    elif args.mode == 'Test':
        generator.process_directory(args.test_folder_source, args.test_folder_des)

    print('============= File conversion completed ==============')
