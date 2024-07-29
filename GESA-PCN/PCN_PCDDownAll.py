"""
Author: Most Husne Jahan
August -2023

To downsample PCN Dataset:

python PCN_PCDDownAll.py

The PCN Dataset is a widely used point cloud completion benchmark that contains 30974 models of 8 categories.

It contains 28974 training samples while each complete sample corresponds to 8 viewpoint partial scans, 800 validation samples, and 1200 testing samples. Each ground-truth point cloud contains 16384 points, which are evenly sampled from the shape surface.

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

import argparse
import os
import open3d as o3d
import numpy as np

class PointCloudProcessor:
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

    def read_pcd(self, filename):
        pcd = o3d.io.read_point_cloud(filename)
        points = np.array(pcd.points)
        return points, pcd

    def save_pcd(self, filename, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd)

    def fps(self, ptcloud, n_points):
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
        return ptcloud

    def downsample(self):
        if not os.path.exists(self.dest):
            os.makedirs(self.dest, exist_ok=True)

        for root, folders, files in os.walk(self.source):
            for folder in folders:
                full_path = os.path.join(self.dest, folder)
                if not os.path.exists(full_path):
                    os.mkdir(full_path)

        for fname in os.listdir(self.source):
            folder_path = os.path.join(self.source, fname)
            out_folder = os.path.join(self.dest, fname)

            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    root, ext = os.path.splitext(file_path)
                    if ext in ['.pcd']:
                        points, pcd = self.read_pcd(file_path)
                        pointcloud = self.fps(points, 2048)
                        out_path = os.path.join(out_folder, file_name)
                        self.save_pcd(out_path, pointcloud)
                        print(f"Saved file {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_complete_source', type=str, default='ShapeNetCompletion/train/complete', help='train complete source')
    parser.add_argument('--train_complete_dest', type=str, default='ShapeNetCompletion2048/train/complete', help='train complete destination')
    parser.add_argument('--val_complete_source', type=str, default='ShapeNetCompletion/val/complete', help='val complete source')
    parser.add_argument('--val_complete_dest', type=str, default='ShapeNetCompletion2048/val/complete', help='val complete destination')
    parser.add_argument('--test_complete_source', type=str, default='ShapeNetCompletion/test/complete', help='test complete source')
    parser.add_argument('--test_complete_dest', type=str, default='ShapeNetCompletion2048/test/complete', help='test complete destination')
    parser.add_argument('--mode', type=str, default='Train', help='Train or Val or Test')
    args = parser.parse_args()

    if args.mode == 'Train':
        processor = PointCloudProcessor(args.train_complete_source, args.train_complete_dest)
    elif args.mode == 'Val':
        processor = PointCloudProcessor(args.val_complete_source, args.val_complete_dest)
    else:
        processor = PointCloudProcessor(args.test_complete_source, args.test_complete_dest)

    processor.downsample()
    print('============= File conversion completed ==============')
