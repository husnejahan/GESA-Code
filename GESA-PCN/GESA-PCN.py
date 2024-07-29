"""
Author: Most Husne Jahan
August -2023
GESA-PCN.py

Generate GESA-PCN Dataset: Apply Monotonic attack (packet loss) to create our own partial point cloud for training, validation set.
"""

import argparse
import os
import open3d as o3d
import numpy as np
from pathlib import Path
from GESA import PointCloudAttack

class GESAPCNDatasetGenerator:
    def __init__(self, train_source, train_dest, val_source, val_dest, test_source, test_dest, mode):
        self.train_source = train_source
        self.train_dest = train_dest
        self.val_source = val_source
        self.val_dest = val_dest
        self.test_source = test_source
        self.test_dest = test_dest
        self.mode = mode
        self.attack_model = PointCloudAttack()

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

    def attack(self, complete_points_path, out_path, packet_loss):
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        for root, folders, files in os.walk(complete_points_path):   
            for folder in folders:
                full_path = os.path.join(out_path, folder)
                if not os.path.exists(full_path):
                    os.mkdir(full_path)

        for fname in os.listdir(complete_points_path):
            folder_path = os.path.join(complete_points_path, fname)
            out_folder = os.path.join(out_path, fname)

            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    points, pcd = self.read_pcd(file_path)
                    points_fps = self.attack_model.fps(points, 2048)
                    pointcloud = self.attack_model.monotonic_attack(points_fps, packet_loss)
                    out_path = os.path.join(out_folder, file_name)
                    self.save_pcd(out_path, pointcloud)
                    print(f"Saved file {out_path}")

    def generate(self):
        packet_loss_total = self.attack_model.packet_loss()
        packet_loss = round(packet_loss_total[3])

        if self.mode == 'Train':
            self.attack(self.train_source, f"{self.train_dest}{packet_loss}", packet_loss)
        elif self.mode == 'Val':    
            self.attack(self.val_source, f"{self.val_dest}{packet_loss}", packet_loss)
        else:        
            self.attack(self.test_source, f"{self.test_dest}{packet_loss}", packet_loss)

        print('File conversion completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder_source', type=str, default='GESA-PCN/train/GT', help='train source')
    parser.add_argument('--train_folder_dest', type=str, default='GESA-PCN/train/attack', help='train destination')
    parser.add_argument('--val_folder_source', type=str, default='GESA-PCN/val/GT', help='val source')
    parser.add_argument('--val_folder_dest', type=str, default='GESA-PCN/val/attack', help='val destination')
    parser.add_argument('--test_folder_source', type=str, default='GESA-PCN/test/GT', help='test source')
    parser.add_argument('--test_folder_dest', type=str, default='GESA-PCN/test/attack', help='test destination')
    parser.add_argument('--mode', type=str, default='Train', help='Train or Val or Test')

    args = parser.parse_args()

    generator = GESAPCNDatasetGenerator(
        train_source=args.train_folder_source,
        train_dest=args.train_folder_dest,
        val_source=args.val_folder_source,
        val_dest=args.val_folder_dest,
        test_source=args.test_folder_source,
        test_dest=args.test_folder_dest,
        mode=args.mode
    )

    generator.generate()
