# import open3d as o3d
# import numpy as np
# import os
# from natsort import natsorted 
# from utils.pose import read_poses_file
# import sys

# def preprocess_kitti(points, z_th=-3.0, min_range=2.5):
#     # filter the outliers
#     # 去掉z轴值小于z_th的点，以及和原点间距离小于min_range的点
#     z = points[:, 2]
#     points = points[z > z_th]
#     points = points[np.linalg.norm(points, axis=1) >= min_range]
#     return points

# def read_point_cloud(filename: str):
#     # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
#     if ".bin" in filename:
#         points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
#     elif ".ply" in filename or ".pcd" in filename:
#         pc_load = o3d.io.read_point_cloud(filename)
#         points = np.asarray(pc_load.points, dtype=np.float64)
#     else:
#         sys.exit(
#             "The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)"
#         )
#     preprocessed_points = preprocess_kitti(
#         points, -10.0, 1.5
#     )
#     pc_out = o3d.geometry.PointCloud()
#     pc_out.points = o3d.utility.Vector3dVector(preprocessed_points) # Vector3dVector is faster for np.float64 
#     return pc_out


# if __name__ == "__main__":
#     pc_path = "/home/cy/NeRF/shine_mapping_input/Map_pointcloud_wo_pose"
#     pose_file_path = "/home/cy/NeRF/shine_mapping_input/Lidar_pose_kitti.txt"
#     output_pc_path = "/home/cy/NeRF/shine_mapping_input/test_combined_pc.pcd"


#     camera2lidar_matrix = np.array([-0.00113207, -0.0158688, 0.999873, 0,
#             -0.9999999,  -0.000486594, -0.00113994, 0,
#             0.000504622,  -0.999874,  -0.0158682, 0,
#             0, 0, 0, 1]).reshape(4, 4)
#     lidar2camera_matrix = np.linalg.inv(camera2lidar_matrix)

#     total_pc = o3d.geometry.PointCloud()

#     calib = {}
#     calib['Tr'] = np.eye(4)
#     poses = read_poses_file(pose_file_path, calib)

#     pc_filenames = natsorted(os.listdir(pc_path))
#     pose_index = 0

#     for filename in pc_filenames:
#         frame_path = os.path.join(pc_path, filename)
#         frame_pc = read_point_cloud(frame_path)

#         bbx_min = np.array([-50.0, -50.0, -10.0])
#         bbx_max = np.array([50.0, 50.0, 30.0])
#         bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
#         frame_pc = frame_pc.crop(bbx)

#         frame_pc = frame_pc.voxel_down_sample(voxel_size=0.05)

#         # 去掉不能映射到相机图片中的点
#         # 将点从雷达坐标系转到相机坐标系
#         frame_pc_points = np.asarray(frame_pc.points, dtype=np.float64)
#         points3d_lidar = np.asarray(frame_pc.points, dtype=np.float64)
#         # points3d_lidar = frame_pc.clone()
#         points3d_lidar = np.insert(points3d_lidar, 3, 1, axis=1)
#         points3d_camera = lidar2camera_matrix @ points3d_lidar.T
#         H, W, fx, fy, cx, cy, = 512, 640, 863.4241, 863.4171, 640.6808, 518.3392
#         K = np.array([[fx, .0, cx, .0], [.0, fy, cy, .0], [.0, .0, 1.0, .0]]).reshape(3, 4)
#         # 过滤掉相机坐标系内位于相机之后的点
#         tmp_mask = points3d_camera[2, :] > 0.0
#         points3d_camera = points3d_camera[:, tmp_mask]
#         frame_pc_points = frame_pc_points[tmp_mask]
#         # 从相机坐标系映射到uv平面坐标
#         points2d_camera = K @ points3d_camera
#         points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 操作之后points2d_camera维度:[n, 2]
#         # 过滤掉uv平面坐标内在图像外的点
#         tmp_mask = np.logical_and(
#             (points2d_camera[:, 1] < H) & (points2d_camera[:, 1] > 0),
#             (points2d_camera[:, 0] < W) & (points2d_camera[:, 0] > 0)
#         )
#         points2d_camera = points2d_camera[tmp_mask]
#         # points3d_camera = (points3d_camera.T)[tmp_mask] # 操作之后points3d_camera维度: [n, 4]
#         frame_pc_points = frame_pc_points[tmp_mask]
#         frame_pc.points = o3d.utility.Vector3dVector(frame_pc_points)

#         frame_pose = poses[pose_index]
#         # frame_pose_inv = np.linalg.inv(frame_pose)

#         total_pc += frame_pc.transform(frame_pose)
#         pose_index += 1

#     o3d.io.write_point_cloud(output_pc_path, total_pc)



import open3d as o3d
import numpy as np

# 创建一个空的TriangleMesh对象
mesh = o3d.geometry.TriangleMesh()

# 创建一个顶点数组
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0]
])

# 创建一个颜色数组，每个顶点对应一个颜色
colors = np.array([
    [1, 0, 0],  # 红色
    [0, 1, 0],  # 绿色
    [0, 0, 1],  # 蓝色
    [1, 1, 0],  # 黄色
])

# 设置顶点和面
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [1, 2, 3]])

# 设置顶点颜色
mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# 创建Visualizer并显示网格
o3d.visualization.draw_geometries([mesh])
