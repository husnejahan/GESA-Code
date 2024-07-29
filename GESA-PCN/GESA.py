"""
Author: Most Husne Jahan
August -2023

GESA.py
Non-Monotonic Attack randomly drops points from the point cloud; 
Monotonic Attack randomly drops several k-NN clusters from the point cloud.
"""

import os
import numpy as np
import random
import open3d as o3d

class AttackSimulator:
    def __init__(self, packet_loss_level):
        self.packet_loss_level = packet_loss_level

    def gilbert_elliot(self, total_packets, p_good_range, p_bad_range, p_g2b_range, p_b2g_range):
        state = "good"
        received_packets = []
        packet_loss_count = 0
        well_received_count = 0

        for _ in range(total_packets):
            p_good = random.uniform(p_good_range[0], p_good_range[1])
            p_bad = random.uniform(p_bad_range[0], p_bad_range[1])
            p_g2b = random.uniform(p_g2b_range[0], p_g2b_range[1])
            p_b2g = random.uniform(p_b2g_range[0], p_b2g_range[1])

            if state == "good":
                if random.random() <= p_good:
                    received_packets.append("well received (1)")
                    well_received_count += 1
                else:
                    received_packets.append("loss (0)")
                    packet_loss_count += 1
                    if random.random() <= p_g2b:
                        state = "bad"
            elif state == "bad":
                if random.random() <= p_bad:
                    received_packets.append("loss (0)")
                    packet_loss_count += 1
                else:
                    received_packets.append("well received (1)")
                    well_received_count += 1
                    if random.random() <= p_b2g:
                        state = "good"

        return received_packets, packet_loss_count, well_received_count

    def packet_loss(self):
        total_packets = 2048
        Packet_Loss = []
        p_good_range = (0.5, 0.9)
        bad_range = [(0.5, 0.6), (0.5, 0.7), (0.5, 0.8), (0.4, 0.9), (0.3, 0.9)]
        p_g2b_range = (0.1, 0.5)
        p_b2g_range = (0.1, 0.5)

        for i in bad_range:
            p_bad_range = i
            results, loss_count, well_received_count = self.gilbert_elliot(total_packets, p_good_range, p_bad_range, p_g2b_range, p_b2g_range)
            packet_loss_ratio = loss_count / total_packets
            Packet_Loss.append(round(packet_loss_ratio * 100))

        return Packet_Loss

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

    def shuffle_pointcloud(self, pcd):
        idx = np.random.rand(pcd.shape[0], 1).argsort(axis=0)
        return np.take_along_axis(pcd, idx, axis=0)

    def gen_random_cluster_sizes(self, num_clusters, total_cluster_size):
        rand_list = np.random.randint(num_clusters, size=total_cluster_size)
        cluster_size_list = [sum(rand_list == i) for i in range(num_clusters)]
        return cluster_size_list

    def monotonic_attack(self, pointcloud):
        num_points = pointcloud.shape[0]
        packet_loss_level = round(self.packet_loss_level / 10)
        total_cluster_size = 100 * (packet_loss_level + 1)
        num_clusters = 1
        cluster_size_list = self.gen_random_cluster_sizes(num_clusters, total_cluster_size)
        for i in range(num_clusters):
            K = cluster_size_list[i]
            pointcloud = self.shuffle_pointcloud(pointcloud)
            dist = np.sum((pointcloud - pointcloud[:1, :]) ** 2, axis=1, keepdims=True)
            idx = dist.argsort(axis=0)[::-1, :]
            pointcloud = pointcloud[idx, :].squeeze(axis=1)
            pointcloud = pointcloud[K:, :]
        return pointcloud

    def non_monotonic_attack(self, pointcloud):
        num_points = pointcloud.shape[0]
        packet_loss_level = round(self.packet_loss_level / 10)
        loss_point = 100 * (packet_loss_level + 1)
        pointcloud = self.shuffle_pointcloud(pointcloud)
        return pointcloud[loss_point:, :]

    def load_and_save_pointcloud(self, source_path, dest_path, attack_type):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)

        for root, folders, files in os.walk(source_path):
            for folder in folders:
                print(os.path.join(root, folder))
                full_path = os.path.join(dest_path, folder)
                if not os.path.exists(full_path):
                    os.mkdir(full_path)

        for fname in os.listdir(source_path):
            folder_path = os.path.join(source_path, fname)
            out_folder = os.path.join(dest_path, fname)

            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    root, ext = os.path.splitext(file_path)
                    if ext in ['.pcd']:
                        pcd = o3d.io.read_point_cloud(file_path)
                        points = np.asarray(pcd.points)

                        if attack_type == 'monotonic':
                            attacked_points = self.monotonic_attack(points)
                        elif attack_type == 'non_monotonic':
                            attacked_points = self.non_monotonic_attack(points)

                        pcd.points = o3d.utility.Vector3dVector(attacked_points)
                        out_path = os.path.join(out_folder, file_name)
                        o3d.io.write_point_cloud(out_path, pcd)
                        print(f"Saved file {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True, help='Source path for point clouds')
    parser.add_argument('--dest_path', type=str, required=True, help='Destination path for attacked point clouds')
    parser.add_argument('--attack_type', type=str, required=True, choices=['monotonic', 'non_monotonic'], help='Type of attack')
    parser.add_argument('--packet_loss_level', type=int, required=True, help='Packet loss level')

    args = parser.parse_args()
    simulator = AttackSimulator(args.packet_loss_level)
    simulator.load_and_save_pointcloud(args.source_path, args.dest_path, args.attack_type)
    print('============= Attack simulation completed ==============')
