import imageio
import cv2
import open3d as o3d
import numpy as np
import os
from natsort import natsorted 
from utils.pose import read_poses_file
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

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
if __name__ == "__main__":
    pc_path = "/home/cy/NeRF/shine_mapping_input/whole_map_wo_pose"
    pose_file_path = "/home/cy/NeRF/shine_mapping_input/Lidar_pose_kitti.txt"
    image_path = "/home/cy/NeRF/shine_mapping_input/image/resized_rgb"
    output_image_folder_path = "/home/cy/NeRF/shine_mapping_input/pc2image_whole_map4/"

    camera2lidar_matrix = np.array([-0.00113207, -0.0158688, 0.999873, 0.050166,
            -0.9999999,  -0.000486594, -0.00113994, 0.0474116,
            0.000504622,  -0.999874,  -0.0158682, -0.0312415,
            0, 0, 0, 1]).reshape(4, 4)
    # lidar2camera_matrix = np.linalg.inv(camera2lidar_matrix)
    lidar2camera_matrix = np.array([[    -0.0000002596,     -0.8758301735,     -0.0000002596,
              0.0336565562],
        [     0.0000002596,      0.0000002596,     -0.8757055402,
             -0.0295539070],
        [     0.8757054806,      0.0000002596,      0.0000002596,
             -0.0338485502],
        [     0.0000000000,      0.0000000000,      0.0000000000,
              1.0165320635]])

    H, W, fx, fy, cx, cy, = 1024, 1280, 863.4241, 863.4171, 640.6808, 518.3392

    distortion_coeffs = np.array([-0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222])  #k1, k2, p1, p2, k3
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    calib = {}
    calib['Tr'] = np.eye(4)
    poses = read_poses_file(pose_file_path, calib)

    pc_filenames = natsorted(os.listdir(pc_path))
    image_filenames = natsorted(os.listdir(image_path))
    pose_index = 0

    for filename in pc_filenames:
        print(filename)
        frame_path = os.path.join(pc_path, filename)
        frame_pc = read_point_cloud(frame_path)

        bbx_min = np.array([-50.0, -50.0, -10.0])
        bbx_max = np.array([50.0, 50.0, 30.0])
        bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
        frame_pc = frame_pc.crop(bbx)

        frame_pc = frame_pc.voxel_down_sample(voxel_size=0.05)

        # 去掉不能映射到相机图片中的点
        # 将点从雷达坐标系转到相机坐标系
        # frame_pc_points = np.asarray(frame_pc.points, dtype=np.float64)
        points3d_lidar = np.asarray(frame_pc.points, dtype=np.float64)
        # points3d_lidar = frame_pc.clone()
        points3d_lidar = np.insert(points3d_lidar, 3, 1, axis=1)
        points3d_camera = lidar2camera_matrix @ points3d_lidar.T
        H, W, fx, fy, cx, cy, = 1024, 1280, 863.4241, 863.4171, 640.6808, 518.3392
        K = np.array([[fx, .0, cx, .0], [.0, fy, cy, .0], [.0, .0, 1.0, .0]]).reshape(3, 4)
        # 过滤掉相机坐标系内位于相机之后的点
        tmp_mask = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, tmp_mask]
        # frame_pc_points = frame_pc_points[tmp_mask]
        # 从相机坐标系映射到uv平面坐标
        points2d_camera = K @ points3d_camera
        points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 操作之后points2d_camera维度:[n, 2]
        # 过滤掉uv平面坐标内在图像外的点
        # TODO H change with W
        tmp_mask = np.logical_and(
            (points2d_camera[:, 1] < H) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < W) & (points2d_camera[:, 0] > 0)
        )
        points2d_camera = points2d_camera[tmp_mask]
        # points3d_camera = (points3d_camera.T)[tmp_mask] # 操作之后points3d_camera维度: [n, 4]
        # frame_pc_points = frame_pc_points[tmp_mask]
        # frame_pc.points = o3d.utility.Vector3dVector(frame_pc_points)
        image_frame_path = os.path.join(image_path, image_filenames[pose_index])
        frame_image = cv2.imread(image_frame_path)
        # frame_image = cv2.undistort(frame_image, camera_matrix, distortion_coeffs)

        white_image = np.ones((H, W, 3), dtype=np.uint8) * 255

        points_int = points2d_camera.astype(int)
        pixel_colors = frame_image[points_int[:, 1], points_int[:, 0]]
        white_image[points_int[:, 1], points_int[:, 0]] = pixel_colors
        cv2.imwrite(output_image_folder_path + filename + ".png", white_image)


        # points2d_camera = points2d_camera.astype(np.int32)
        # # 将指定坐标处的像素点调整为白色
        # for point in points2d_camera:
        #     x, y = point  # 提取坐标
        #     black_image[y, x] = [255, 255, 255]  # 设置像素点颜色为白色
        # cv2.imwrite(output_image_folder_path + filename + ".png", black_image)

        # frame_pose = poses[pose_index]
        # frame_pose_inv = np.linalg.inv(frame_pose)

        # total_pc += frame_pc.transform(frame_pose)
        pose_index += 1



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

