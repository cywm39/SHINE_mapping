import imageio
import cv2
import open3d as o3d
import numpy as np
import os
import trimesh
from natsort import natsorted 
from utils.pose import read_poses_file
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from model.feature_octree import FeatureOctree
from utils.config import SHINEConfig
from utils.visualizer import MapVisualizer, random_color_table
from model.color_decoder import ColorDecoder
from model.sdf_decoder import SDFDecoder
from utils.mesher import Mesher
from utils.tools import *
# import matplotlib.pyplot as plt


def preprocess_kitti(points, z_th=-3.0, min_range=2.5):
    # filter the outliers
    # 去掉z轴值小于z_th的点，以及和原点间距离小于min_range的点
    z = points[:, 2]
    points = points[z > z_th]
    points = points[np.linalg.norm(points, axis=1) >= min_range]
    return points

def read_point_cloud(filename: str):
    # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
    if ".bin" in filename:
        points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
    elif ".ply" in filename or ".pcd" in filename:
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points, dtype=np.float64)
    else:
        sys.exit(
            "The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)"
        )
    preprocessed_points = preprocess_kitti(
        points, -10.0, 1.5
    )
    pc_out = o3d.geometry.PointCloud()
    pc_out.points = o3d.utility.Vector3dVector(preprocessed_points) # Vector3dVector is faster for np.float64 
    return pc_out

def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


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



# import open3d as o3d
# import numpy as np

# # 创建一个空的TriangleMesh对象
# mesh = o3d.geometry.TriangleMesh()

# # 创建一个顶点数组
# vertices = np.array([
#     [0, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 1, 0]
# ])

# # 创建一个颜色数组，每个顶点对应一个颜色
# colors = np.array([
#     [1, 0, 0],  # 红色
#     [0, 1, 0],  # 绿色
#     [0, 0, 1],  # 蓝色
#     [1, 1, 0],  # 黄色
# ])

# # 设置顶点和面
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [1, 2, 3]])

# # 设置顶点颜色
# mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# # 创建Visualizer并显示网格
# o3d.visualization.draw_geometries([mesh])


# 点云映射到图片上
# if __name__ == "__main__":
#     pc_path = "/home/cy/NeRF/shine_mapping_input/whole_map_wo_pose"
#     pose_file_path = "/home/cy/NeRF/shine_mapping_input/Lidar_pose_kitti.txt"
#     image_path = "/home/cy/NeRF/shine_mapping_input/image/resized_rgb"
#     output_image_folder_path = "/home/cy/NeRF/shine_mapping_input/pc2image_whole_map4/"

#     camera2lidar_matrix = np.array([-0.00113207, -0.0158688, 0.999873, 0.050166,
#             -0.9999999,  -0.000486594, -0.00113994, 0.0474116,
#             0.000504622,  -0.999874,  -0.0158682, -0.0312415,
#             0, 0, 0, 1]).reshape(4, 4)
#     # lidar2camera_matrix = np.linalg.inv(camera2lidar_matrix)
#     lidar2camera_matrix = np.array([[    -0.0000002596,     -0.8758301735,     -0.0000002596,
#               0.0336565562],
#         [     0.0000002596,      0.0000002596,     -0.8757055402,
#              -0.0295539070],
#         [     0.8757054806,      0.0000002596,      0.0000002596,
#              -0.0338485502],
#         [     0.0000000000,      0.0000000000,      0.0000000000,
#               1.0165320635]])

#     H, W, fx, fy, cx, cy, = 1024, 1280, 863.4241, 863.4171, 640.6808, 518.3392

#     distortion_coeffs = np.array([-0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222])  #k1, k2, p1, p2, k3
#     camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

#     calib = {}
#     calib['Tr'] = np.eye(4)
#     poses = read_poses_file(pose_file_path, calib)

#     pc_filenames = natsorted(os.listdir(pc_path))
#     image_filenames = natsorted(os.listdir(image_path))
#     pose_index = 0

#     for filename in pc_filenames:
#         print(filename)
#         frame_path = os.path.join(pc_path, filename)
#         frame_pc = read_point_cloud(frame_path)

#         bbx_min = np.array([-50.0, -50.0, -10.0])
#         bbx_max = np.array([50.0, 50.0, 30.0])
#         bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
#         frame_pc = frame_pc.crop(bbx)

#         frame_pc = frame_pc.voxel_down_sample(voxel_size=0.05)

#         # 去掉不能映射到相机图片中的点
#         # 将点从雷达坐标系转到相机坐标系
#         # frame_pc_points = np.asarray(frame_pc.points, dtype=np.float64)
#         points3d_lidar = np.asarray(frame_pc.points, dtype=np.float64)
#         # points3d_lidar = frame_pc.clone()
#         points3d_lidar = np.insert(points3d_lidar, 3, 1, axis=1)
#         points3d_camera = lidar2camera_matrix @ points3d_lidar.T
#         H, W, fx, fy, cx, cy, = 1024, 1280, 863.4241, 863.4171, 640.6808, 518.3392
#         K = np.array([[fx, .0, cx, .0], [.0, fy, cy, .0], [.0, .0, 1.0, .0]]).reshape(3, 4)
#         # 过滤掉相机坐标系内位于相机之后的点
#         tmp_mask = points3d_camera[2, :] > 0.0
#         points3d_camera = points3d_camera[:, tmp_mask]
#         # frame_pc_points = frame_pc_points[tmp_mask]
#         # 从相机坐标系映射到uv平面坐标
#         points2d_camera = K @ points3d_camera
#         points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 操作之后points2d_camera维度:[n, 2]
#         # 过滤掉uv平面坐标内在图像外的点
#         # TODO H change with W
#         tmp_mask = np.logical_and(
#             (points2d_camera[:, 1] < H) & (points2d_camera[:, 1] > 0),
#             (points2d_camera[:, 0] < W) & (points2d_camera[:, 0] > 0)
#         )
#         points2d_camera = points2d_camera[tmp_mask]
#         # points3d_camera = (points3d_camera.T)[tmp_mask] # 操作之后points3d_camera维度: [n, 4]
#         # frame_pc_points = frame_pc_points[tmp_mask]
#         # frame_pc.points = o3d.utility.Vector3dVector(frame_pc_points)
#         image_frame_path = os.path.join(image_path, image_filenames[pose_index])
#         frame_image = cv2.imread(image_frame_path)
#         # frame_image = cv2.undistort(frame_image, camera_matrix, distortion_coeffs)

#         white_image = np.ones((H, W, 3), dtype=np.uint8) * 255

#         points_int = points2d_camera.astype(int)
#         pixel_colors = frame_image[points_int[:, 1], points_int[:, 0]]
#         white_image[points_int[:, 1], points_int[:, 0]] = pixel_colors
#         cv2.imwrite(output_image_folder_path + filename + ".png", white_image)


#         # points2d_camera = points2d_camera.astype(np.int32)
#         # # 将指定坐标处的像素点调整为白色
#         # for point in points2d_camera:
#         #     x, y = point  # 提取坐标
#         #     black_image[y, x] = [255, 255, 255]  # 设置像素点颜色为白色
#         # cv2.imwrite(output_image_folder_path + filename + ".png", black_image)

#         # frame_pose = poses[pose_index]
#         # frame_pose_inv = np.linalg.inv(frame_pose)

#         # total_pc += frame_pc.transform(frame_pose)
#         pose_index += 1



# if __name__ == "__main__":
#     pc_path = "/home/cy/NeRF/shine_mapping_input/Map_pointcloud_wo_pose"
#     pose_file_path = "/home/cy/NeRF/shine_mapping_input/Lidar_pose_kitti.txt"
#     output_pc_folder_path = "/home/cy/NeRF/shine_mapping_input/Map_pointcloud_wo_pose_combine7/"

#     camera2lidar_matrix = np.array([-0.00113207, -0.0158688, 0.999873, 0,
#             -0.9999999,  -0.000486594, -0.00113994, 0,
#             0.000504622,  -0.999874,  -0.0158682, 0,
#             0, 0, 0, 1]).reshape(4, 4)
#     lidar2camera_matrix = np.linalg.inv(camera2lidar_matrix)

#     calib = {}
#     calib['Tr'] = np.eye(4)
#     poses = read_poses_file(pose_file_path, calib)

#     pc_filenames = natsorted(os.listdir(pc_path))
#     i = 0

#     while(True):
#         if i >= 861:
#             break

#         frame1_path = os.path.join(pc_path, pc_filenames[i])
#         frame_pc1 = o3d.io.read_point_cloud(frame1_path)

#         frame2_path = os.path.join(pc_path, pc_filenames[i + 1])
#         frame_pc2 = o3d.io.read_point_cloud(frame2_path)

#         frame3_path = os.path.join(pc_path, pc_filenames[i + 2])
#         frame_pc3 = o3d.io.read_point_cloud(frame3_path)

#         frame4_path = os.path.join(pc_path, pc_filenames[i + 3])
#         frame_pc4 = o3d.io.read_point_cloud(frame4_path)

#         frame5_path = os.path.join(pc_path, pc_filenames[i + 4])
#         frame_pc5 = o3d.io.read_point_cloud(frame5_path)

#         frame6_path = os.path.join(pc_path, pc_filenames[i + 5])
#         frame_pc6 = o3d.io.read_point_cloud(frame6_path)

#         frame7_path = os.path.join(pc_path, pc_filenames[i + 6])
#         frame_pc7 = o3d.io.read_point_cloud(frame7_path)

#         frame_pc1 = frame_pc1.transform(poses[i])
#         frame_pc2 = frame_pc2.transform(poses[i + 1])
#         frame_pc3 = frame_pc3.transform(poses[i + 2])
#         frame_pc4 = frame_pc4.transform(poses[i + 3])
#         frame_pc5 = frame_pc5.transform(poses[i + 4])
#         frame_pc6 = frame_pc6.transform(poses[i + 5])
#         frame_pc7 = frame_pc7.transform(poses[i + 6])

#         total_frame = frame_pc1 + frame_pc2 + frame_pc3 + frame_pc4 + frame_pc5 + frame_pc6 + frame_pc7
#         total_frame = total_frame.transform(np.linalg.inv(poses[i]))
#         o3d.io.write_point_cloud(output_pc_folder_path + pc_filenames[i], total_frame)

#         i += 7


# if __name__ == "__main__":
#     input_file_path = '/home/cy/NeRF/shine_mapping_input/Lidar_pose_kitti.txt'
#     # 输出文件的路径
#     output_file_path = '/home/cy/NeRF/shine_mapping_input/Lidar_pose_kitti_combine7.txt'

#     # 打开输入文件和输出文件
#     with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
#         # 逐行读取输入文件
#         lines = input_file.readlines()
        
#         # 遍历每一行，从0开始计数
#         for i, line in enumerate(lines):
#             # 判断是否为奇数行
#             if i % 7 == 0:
#                 # 将奇数行写入输出文件
#                 output_file.write(line)

# import os
# import shutil

# source_folder = '/home/cy/NeRF/shine_mapping_input/image/resized_rgb'  # 修改为你的源文件夹路径
# destination_folder = '/home/cy/NeRF/shine_mapping_input/image/resized_rgb_combine7'  # 修改为你的目标文件夹路径

# # 获取源文件夹中的所有文件
# files = natsorted(os.listdir(source_folder))

# # 筛选奇数张图片并复制到目标文件夹
# for i, filename in enumerate(files):
#     if i % 7 == 0:  # 仅复制奇数张图片，i从0开始计数
#         source_path = os.path.join(source_folder, filename)
#         destination_path = os.path.join(destination_folder, filename)
#         shutil.copy(source_path, destination_path)

# print("复制完成")


# if __name__ == "__main__":
#     source_pc_path = "/home/cy/NeRF/shine_mapping_input/merged_cloud.pcd"
#     pose_file_path = "/home/cy/NeRF/shine_mapping_input/Lidar_pose_kitti.txt"
#     pc_name_path = "/home/cy/NeRF/shine_mapping_input/Map_pointcloud_wo_pose"
#     output_path = "/home/cy/NeRF/shine_mapping_input/whole_map_wo_pose/"

#     pc_filenames = natsorted(os.listdir(pc_name_path))

#     calib = {}
#     calib['Tr'] = np.eye(4)
#     poses = read_poses_file(pose_file_path, calib)

#     index = 0

#     for filename in pc_filenames:
#         source_pc = o3d.io.read_point_cloud(source_pc_path)
#         pose = poses[index]
#         source_pc = source_pc.transform(np.linalg.inv(pose))
#         o3d.io.write_point_cloud(output_path + filename, source_pc)
#         index += 1

# # retain_graph测试程序        
# if __name__ == "__main__":
#     torch.autograd.set_detect_anomaly(True)
#     lidar2camera_matrix = nn.Parameter(torch.rand([4,4], requires_grad=True))
#     for frame_id in tqdm(range(100)):
#         coord = lidar2camera_matrix @ torch.rand(4,4)
#         opt = optim.Adam([{'params': [lidar2camera_matrix], 'lr': 1e-3, 'weight_decay': 0}], betas=(0.9,0.99), eps = 1e-15) 
        
#         for iter in tqdm(range(50)):
#             index = torch.randint(0, 4, (1,))
#             y = coord[index]
#             loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
#             loss = loss_bce(y, torch.tensor([[1.,1.,1.,1.]]))

#             opt.zero_grad()
#             loss.backward(retain_graph=True)
#             # loss.backward()
#             opt.step()


# if __name__ == "__main__":
#     mesh_tmp = o3d.io.read_triangle_mesh("/home/cy/Test/r3live_mesh_frame_863.ply")
#     o3d.visualization.draw_geometries([mesh_tmp])
#     mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_tmp)
#     print(type(mesh))
#     # torus = o3d.geometry.TriangleMesh.create_torus().translate([0, 0, 2])
#     # torus = o3d.t.geometry.TriangleMesh.from_legacy(torus)

#     scene = o3d.t.geometry.RaycastingScene()
#     scene.add_triangles(mesh)

#     # c2w = torch.tensor([[-1.9808e-03, -1.0000e+00, 1.1601e-03, 7.0236e-03],
#     #     [-1.3469e-02, -1.1334e-03, -9.9991e-01, -7.0353e-02],
#     #     [9.9991e-01, -1.9963e-03, -1.3467e-02, -3.2179e-02],
#     #     [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])
#     # c2w = torch.tensor([9.999994249509230881e-01, -6.419753489684488969e-04, -8.590491688647965704e-04,-4.052349999999999702e-02,
#     #                     6.440357145296518593e-04, 9.999969118786344868e-01, 2.400302312818621692e-03, -3.984349999999999697e-02,
#     #                     8.575055811018025788e-04, -2.400854190872278061e-03, 9.999967502863859048e-01, 1.830839999999999898e-02,
#     #                     0,0,0,1]).reshape(4,4)
#     c2w = torch.tensor([6.095023248645019542e-01, 7.728227619515996016e-01, -1.767820539372378374e-01, 6.002030000000000420e-01,
#                         -7.351278807392950254e-01, 6.344284017162953315e-01, 2.389301196070927058e-01, 2.378190000000000026e-01,
#                         2.968061908797051673e-01, -1.567104671705508989e-02, 9.548091449867203151e-01, 1.501360000000000028e+00,
#                         0,0,0,1]).reshape(4,4)

#     H, W, fx, fy, cx, cy, = 1024, 1280, 863.4241, 863.4171, 640.6808, 518.3392

#     rays_o, rays_d = get_rays(H, W, fx, fy, cx, cy, c2w, "cpu")
#     # rays_d = rays_d * 10000
#     rays = torch.cat([rays_o, rays_d], -1)

#     print(rays.shape)

#     rays = o3d.core.Tensor(rays.numpy())

#     ans = scene.cast_rays(rays)

#     hit = ans['t_hit'].isfinite()
#     points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
#     print(type(points))
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points.numpy())
#     o3d.io.write_point_cloud("test.pcd", point_cloud)
#     # plt.imsave("test.jpg",ans['t_hit'].numpy())
#     # plt.imsave("test.jpg",np.abs(ans['primitive_normals'].numpy()))
#     # plt.imsave("test.jpg", ans['geometry_ids'].numpy(), vmax=3)
#     print("fin")

#     # tmp = torch.zeros(100,3)
#     # rays_o = rays_o.reshape(-1, 3)
#     # rays_d = rays_d.reshape(-1, 3)
#     # rays = torch.cat([rays_o, rays_d])
#     # rays = torch.cat([rays, tmp])
#     # point_cloud = o3d.geometry.PointCloud()
#     # point_cloud.points = o3d.utility.Vector3dVector(rays.numpy())
#     # o3d.io.write_point_cloud("test.pcd", point_cloud)


# if __name__ == "__main__":
    # config = SHINEConfig()
    # config.load(config_file_path)
    # sdf_octree = FeatureOctree(config, is_color=False)
    # loaded_model = torch.load(load_model_path)
    # if 'sdf_octree' in loaded_model.keys(): # also load the feature octree  
    #     sdf_octree = loaded_model["sdf_octree"]
    #     sdf_octree.print_detail()
    # # visualize the octree (it is a bit slow and memory intensive for the visualization)
    # vis_octree = True
    # if vis_octree: 
    #     vis_list = [] # create a list of bbx for the octree nodes
    #     for l in range(config.tree_level_feat):
    #         nodes_coord = sdf_octree.get_octree_nodes(config.tree_level_world-l)/config.scale
    #         box_size = np.ones(3) * config.leaf_vox_size * (2**l)
    #         for node_coord in nodes_coord:
    #             node_box = o3d.geometry.AxisAlignedBoundingBox(node_coord-0.5*box_size, node_coord+0.5*box_size)
    #             node_box.color = random_color_table[l]
    #             vis_list.append(node_box)
    #     o3d.visualization.draw_geometries(vis_list)

# import numpy as np
# from scipy.spatial.transform import Rotation as R

# def create_transformation_matrix(translation, rotation):
#     # 创建变换矩阵
#     matrix = np.eye(4)
#     matrix[:3, 3] = translation
#     matrix[:3, :3] = R.from_euler('xyz', rotation, degrees=True).as_matrix()
#     return matrix

# # 相机外部参数
# camera_translation = np.array([3.5, 0.2, 2.55])
# camera_rotation = np.array([0.0, 20.0, 0.0])

# # 雷达外部参数
# lidar_translation = np.array([3.5, -0.2, 2.55])
# lidar_rotation = np.array([0.0, 0.0, 0.0])

# # 创建相机和雷达的变换矩阵
# camera_matrix = create_transformation_matrix(camera_translation, camera_rotation)
# lidar_matrix = create_transformation_matrix(lidar_translation, lidar_rotation)

# # 计算相机到雷达的相对变换矩阵
# camera_to_lidar_matrix = np.linalg.inv(lidar_matrix) @ camera_matrix

# print("Camera to Lidar Transformation Matrix:")
# print(camera_to_lidar_matrix)




# import open3d as o3d
# import numpy as np
# from copy import deepcopy as dc
# from matplotlib import pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.image as mpimg
# from scipy.spatial.transform import Rotation as R


# def projection(points, camera_scan, intrinsics, extrinsics, filter_outliers=True, vis=True):
#     trans_lidar_to_camera = extrinsics

#     points3d_lidar = points
#     points3d_lidar = np.insert(points3d_lidar,3,1,axis=1)

#     points3d_camera = trans_lidar_to_camera @ points3d_lidar.T

#     K = intrinsics

#     inliner_indices_arr = np.arange(points3d_camera.shape[1])
#     if filter_outliers:
#         condition = points3d_camera[2, :] > 0.0
#         points3d_camera = points3d_camera[:, condition]
#         inliner_indices_arr = inliner_indices_arr[condition]

#     points2d_camera = K @ points3d_camera
#     points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T
#     image_h, image_w, _ = camera_scan.shape

#     if filter_outliers:
#         condition = np.logical_and(
#         (points2d_camera[:, 1] < image_h) & (points2d_camera[:, 1] > 0),
#         (points2d_camera[:, 0] < image_w) & (points2d_camera[:, 0] > 0))
#         points2d_camera = points2d_camera[condition]
#         points3d_camera = (points3d_camera.T)[condition]
#         inliner_indices_arr = inliner_indices_arr[condition]
#     else:
#         points3d_camera = points3d_camera.T
#         if(len(points2d_camera) == 0):
#             print("no point on the image")
#         return
#     if(vis):
#         plt.imshow(camera_scan)
#         distances = np.sqrt(np.sum(np.square(points3d_camera), axis=-1))
#         colors = cm.jet(distances / np.max(distances))
#         plt.gca().scatter(points2d_camera[:, 0], points2d_camera[:, 1], color=colors, s=1)
#         plt.show()
# def main():
#     cloud = o3d.io.read_point_cloud("pcd_path")
#     cloud.voxel_down_sample(4)
#     camera = mpimg.imread("image_path")
#     intrinsics = np.array( ### edit
#         [[400., 0.0, 400., 0],
#         [0.0, 400., 300., 0],
#         [0.0, 0.0, 1.0, 0]])
#     extrinsics = np.array(### edit
#         [ 0.93969262, 0., -0.34202014, 0.,
#                       0.,  1., 0., -0.4,
#                       0.34202014,  0.,  0.93969262, 0.,
#                       0., 0., 0., 1.]).reshape(4,4)


#     projection(np.array(cloud.points), camera, intrinsics, extrinsics, filter_outliers=True, vis=True)

# main()


import numpy as np
from scipy.spatial.transform import Rotation as R

def create_transformation_matrix(translation, rotation):
    # 创建变换矩阵
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    matrix[:3, :3] = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    return matrix

# 相机外部参数
camera_translation = np.array([3.5, 0.2, 2.55])
camera_rotation = np.array([20.0, 0.0, 0.0])

# 雷达外部参数
lidar_translation = np.array([3.5, -0.2, 2.55])
lidar_rotation = np.array([0.0, 0.0, 0.0])

# 创建相机和雷达的变换矩阵
camera_matrix = create_transformation_matrix(camera_translation, camera_rotation)
lidar_matrix = create_transformation_matrix(lidar_translation, lidar_rotation)

# 计算相机到雷达的相对变换矩阵
camera_to_lidar_matrix = np.linalg.inv(camera_matrix) @ lidar_matrix

print("Camera to Lidar Transformation Matrix:")
print(camera_to_lidar_matrix)
