"""

Author: Most Husne Jahan
August -2023

File: Gilbert-Attack.py

"Non-Monotonic Attack" randomly drops points from the point cloud; 
"Monotonic Attack" randomly drops several k-NN clusters from the point cloud;

Generate GESA Dataset

"""


import os
import numpy as np
import itertools
import math, random
random.seed = 42

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import scipy.spatial.distance
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
#from path import Path
import random
import open3d as o3d

################ LOSS CALCULATION ALGORITHOM #########################################

def gilbert_elliot(total_packets, p_good_range, p_bad_range, p_g2b_range, p_b2g_range):
    state = "good"  # Start in the "good" state
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


def packet_loss(): 

    # Lets assume Raw point cloud N =2048
    total_packets = 2048
    Packet_Loss =[]
    # p_good_range = (0.89, 0.9)
    p_good_range = (0.5, 0.9)
    # p_good_range = (0.8, 0.9)

    p_bad_range = (0.5, 0.99)
    #bad_range = [(0.5, 0.6),(0.5, 0.7),(0.5, 0.8),(0.4, 0.9),(0.3, 0.9)]
    bad_range = [(0.5, 0.6),(0.5, 0.7),(0.5, 0.8),(0.4, 0.9),(0.3, 0.9)]
    p_g2b_range = (0.1, 0.5)
    p_b2g_range = (0.1, 0.5)

    for i in bad_range:
        p_bad_range = i
        results, loss_count, well_received_count = gilbert_elliot(total_packets, p_good_range, p_bad_range, p_g2b_range, p_b2g_range)

        # Calculate and print the ratios
        packet_loss_ratio = loss_count / total_packets
        well_received_ratio = well_received_count / total_packets
        # print(f"\nBad_range #: {p_bad_range}")
        # print(f"Packet loss ratio: {packet_loss_ratio:.2f}")

        Packet_Loss.append(round(packet_loss_ratio*100))
    
        # print(f"\n Packet_Loss #: {Packet_Loss}")

    return Packet_Loss  

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


def _shuffle_pointcloud(pcd):
    """
    Shuffle the points
    :param pcd: input point cloud
    :return: shuffled point clouds
    """
    idx = np.random.rand(pcd.shape[0], 1).argsort(axis=0)
    return np.take_along_axis(pcd, idx, axis=0)

#######################################################

"""
"Monotonic Attack" randomly drops several k-NN clusters from the point cloud;
6-cluster, 2048 points, random cluster size

"""

#########################################################

def _gen_random_cluster_sizes(num_clusters, total_cluster_size):
    """
    Generate random cluster sizes
    :param num_clusters: number of clusters
    :param total_cluster_size: total size of all clusters
    :return: a list of each cluster size
    """
    rand_list = np.random.randint(num_clusters, size=total_cluster_size)
    cluster_size_list = [sum(rand_list == i) for i in range(num_clusters)]
    return cluster_size_list

def Monotonic_Attack(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    Packet_Loss_level= round(level/10)
    total_cluster_size = 100 * (Packet_Loss_level + 1)
    # 8cluster, 2048 points, random cluster size
    # num_clusters = np.random.randint(1, 5)
    num_clusters = 1
    # print(f"\n Total Cluster #: {num_clusters}")
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    for i in range(num_clusters):
        K = cluster_size_list[i]
        pointcloud = _shuffle_pointcloud(pointcloud)
        dist = np.sum((pointcloud - pointcloud[:1, :]) ** 2, axis=1, keepdims=True)
        idx = dist.argsort(axis=0)[::-1, :]
        pointcloud = np.take_along_axis(pointcloud, idx, axis=0)
        num_points -= K
        pointcloud = pointcloud[:num_points, :]
    return pointcloud

#########################################################

"""
"Non-Monotonic Attack" randomly drops points from the point cloud; 
Packet_Loss #: [28, 27, 31, 30, 29]
 
"""

#########################################################

def NonMonotonic_Attack(pointcloud,level):
    """
    Drop random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    #drop_rate = [0.15, 0.25, 0.35, 0.45, 0.50][level]
    drop_rate = level/100
    num_points = pointcloud.shape[0]
    pointcloud = _shuffle_pointcloud(pointcloud)
    pointcloud = pointcloud[:int(num_points * (1 - drop_rate)), :]
    return pointcloud
#########################################################
