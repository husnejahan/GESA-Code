'''
python3 -m venv my-env
source my-env/bin/activate
pip3 install -r requirements.txt
pip3 install open3d

python3 dataviz_PCN.py



'''

import open3d as o3d
import numpy as np
import os

# Visualize GT

# cloud = o3d.io.read_point_cloud('./ShapeNetCompletion2048/train/partial/02691156/1a04e3eab45ca15dd86060f189eb133/01.pcd')
cloud = o3d.io.read_point_cloud('./ShapeNetCompletion2048/train/partial/02691156/1a04e3eab45ca15dd86060f189eb133/02.pcd')
points = np.array(cloud.points)
print(points.shape)
# cloud = o3d.io.read_point_cloud('./ShapeNetCompletion/train/complete/02691156/1a74b169a76e651ebc0909d98a1ff2b4.pcd')
# cloud = o3d.io.read_point_cloud('./ShapeNetCompletion/train/complete/02958343/1a83a525ae32afcf622ac2f50deaba1f.pcd')
# cloud = o3d.io.read_point_cloud('./ShapeNetCompletion/train/complete/03001627/1b67a3a1101a9acb905477d2a8504646.pcd')
# cloud = o3d.io.read_point_cloud('./ShapeNetCompletion/train/complete/04379243/1a15e651e26622b0e5c7ea227b17d897.pcd')

# cloud = o3d.io.read_point_cloud('./ShapeNetCompletion/train/complete/04530566/ffacadade68cec7b926a1ee5a429907.pcd')
# cloud = o3d.io.read_point_cloud('./ShapeNetCompletion/train/complete/03636649/1b0ecf93a4a36a70f7b783634bf3f92f.pcd')
# o3d.visualization.draw_geometries([cloud])

# Visualize GT

# Visualize Attack


# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN-V1/train/attack/02691156/1a74b169a76e651ebc0909d98a1ff2b4.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN-V2/train/attack/02691156/1a74b169a76e651ebc0909d98a1ff2b4.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN-V3/train/attack/02691156/1a74b169a76e651ebc0909d98a1ff2b4.pcd')

# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack17/02958343/1a83a525ae32afcf622ac2f50deaba1f.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack28/02958343/1a83a525ae32afcf622ac2f50deaba1f.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack46/02958343/1a83a525ae32afcf622ac2f50deaba1f.pcd')


# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack17/03001627/1b67a3a1101a9acb905477d2a8504646.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack28/03001627/1b67a3a1101a9acb905477d2a8504646.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack46/03001627/1b67a3a1101a9acb905477d2a8504646.pcd')


# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack17/04530566/ffacadade68cec7b926a1ee5a429907.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack28/04530566/ffacadade68cec7b926a1ee5a429907.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack46/04530566/ffacadade68cec7b926a1ee5a429907.pcd')

# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack17/03636649/1b0ecf93a4a36a70f7b783634bf3f92f.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack28/03636649/1b0ecf93a4a36a70f7b783634bf3f92f.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack46/03636649/1b0ecf93a4a36a70f7b783634bf3f92f.pcd')

# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack17/04379243/1a15e651e26622b0e5c7ea227b17d897.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack28/04379243/1a15e651e26622b0e5c7ea227b17d897.pcd')
# cloud = o3d.io.read_point_cloud('./GESA-PCA-PCN/train/attack46/04379243/1a15e651e26622b0e5c7ea227b17d897.pcd')






# Visualize Attack

# print(np.asarray(cloud.points))
# print(np.asarray(cloud.colors))
# print(np.asarray(cloud.normals))
o3d.visualization.draw_geometries([cloud])