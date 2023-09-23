import open3d as o3d
import numpy as np
import os
from natsort import natsorted 
from utils.pose import read_poses_file

pc_path = "/home/cy/NeRF/shine_mapping_input/Map_pointcloud_with_pose"
pose_file_path = "/home/cy/NeRF/shine_mapping_input/Lidar_pose_kitti.txt"
output_pc_path = "/home/cy/NeRF/shine_mapping_input/Map_pointcloud_wo_pose/"

calib = {}
calib['Tr'] = np.eye(4)
poses = read_poses_file(pose_file_path, calib)

pc_filenames = natsorted(os.listdir(pc_path))
pose_index = 0

for filename in pc_filenames:
    frame_path = os.path.join(pc_path, filename)
    pc_in = o3d.io.read_point_cloud(frame_path)
    frame_pose = poses[pose_index]
    frame_pose_inv = np.linalg.inv(frame_pose)
    o3d.io.write_point_cloud(output_pc_path + filename, pc_in.transform(frame_pose_inv))
    pose_index += 1