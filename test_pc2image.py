import open3d as o3d
import numpy as np
from copy import deepcopy as dc
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from scipy.spatial.transform import Rotation as R
from utils.pose import read_poses_file


def projection(points, camera_scan, intrinsics, extrinsics, filter_outliers=True, vis=True):
    trans_lidar_to_camera = extrinsics

    c2p = np.array([[0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0,0,0,1]]).reshape(4,4)

    points3d_lidar = points
    points3d_lidar = np.insert(points3d_lidar,3,1,axis=1)
    points3d_lidar = points3d_lidar @ c2p

    points3d_camera = trans_lidar_to_camera @ points3d_lidar.T

    K = intrinsics

    inliner_indices_arr = np.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = K @ points3d_camera
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T
    image_h, image_w, _ = camera_scan.shape

    if filter_outliers:
        condition = np.logical_and(
            (points2d_camera[:, 1] < image_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < image_w) & (points2d_camera[:, 0] > 0))
        points2d_camera = points2d_camera[condition]
        points3d_camera = (points3d_camera.T)[condition]
        inliner_indices_arr = inliner_indices_arr[condition]
    else:
        points3d_camera = points3d_camera.T

    if(len(points2d_camera) == 0):
        print("no point on the image")
        return
    if(vis):
        plt.imshow(camera_scan)
        distances = np.sqrt(np.sum(np.square(points3d_camera), axis=-1))
        colors = cm.jet(distances / 20) #np.max(distances)
        plt.gca().scatter(points2d_camera[:, 0], points2d_camera[:, 1], color=colors, s=1)
        plt.show()


def E_Calc(roll, pitch, yaw, x, y, z):
    yaw = np.radians(yaw)
    roll = np.radians(roll)
    pitch = np.radians(pitch)

    yaw_matrix = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitch_matrix = np.matrix([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])

    roll_matrix = np.matrix([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    rotation_matrix = yaw_matrix * pitch_matrix * roll_matrix

    # translation_matrix = np.matrix([
    # [x],
    # [y],
    # [z]
    # ])
    translation_matrix = np.array([x, y, z])
    # 拼接旋转和平移矩阵并增加0,0,0,1
    extrinsic_matrix = np.hstack([np.array(rotation_matrix), np.array([translation_matrix]).T])
    extrinsic_matrix = np.vstack([np.array(extrinsic_matrix), np.array([[0,0,0,1]]) ])
    return extrinsic_matrix


def main():
    # cloud = o3d.io.read_point_cloud("/home/wuchenyang/NeRF/dataset/carla/2023_11_14/livox_pc/177.505801540.pcd")
    

    cloud = o3d.io.read_point_cloud("/home/wuchenyang/NeRF/dataset/carla/2023_11_14/livox_pc/207.155801982.pcd")
    cloud.voxel_down_sample(4)

    calib = {}
    calib['Tr'] = np.eye(4)
    poses = read_poses_file("/home/wuchenyang/NeRF/dataset/carla/2023_11_14/lidar_odometry_kitti.txt", calib)
        
    
    # L2w = np.array([[-9.99999959e-01, 1.37270522e-07, 2.87085461e-04, -2.20829554e+00],
    #                 [-2.87085462e-04, -8.22540320e-06, -9.99999959e-01, -1.90362036e+02],
    #                 [-1.34909123e-07, -1.00000000e+00, 8.22544227e-06, 2.55187111e+00],
    #                 [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # cloud = cloud.transform(np.linalg.inv(L2w))
    # camera = mpimg.imread("/home/wuchenyang/NeRF/dataset/carla/2023_11_14/image/177.505802.png")
    camera = mpimg.imread("/home/wuchenyang/NeRF/dataset/carla/2023_11_14/image/207.155802.png")
    intrinsics = np.array( ### edit
        [[400., 0.0, 400., 0],
        [0.0, 400., 300., 0],
        [0.0, 0.0, 1.0, 0]])
    # extrinsics = np.array(### edit
    #     [ 0.93969262, 0., -0.34202014, 0.,
    #                   0.,  1., 0., -0.4,
    #                   0.34202014,  0.,  0.93969262, 0.,
    #                   0., 0., 0., 1.]).reshape(4,4)

    # camera_m = E_Calc(0.0, 20.0, 0.0, 3.5, 0.2, 2.55)
    # lidar_m = E_Calc(0.0, 0.0, 0.0, 3.5, -0.2, 2.55)
    # extrinsics = np.linalg.inv(camera_m) @ lidar_m
    # print(extrinsics)


    def create_transformation_matrix(translation, rotation):
        # 创建变换矩阵
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        matrix[:3, :3] = R.from_euler('xyz', rotation, degrees=True).as_matrix()
        return matrix

    # 相机外部参数
    camera_translation = np.array([3.5, 0.2, 2.55]) # 这三个的顺序就是xyz
    camera_rotation = np.array([20.0, 0.0, 0.0]) # 这三个的顺序是pitch在最前面, 后面两个可能是[roll, yaw], 也可能是反过来的

    # 雷达外部参数
    lidar_translation = np.array([3.5, -0.2, 2.55])
    lidar_rotation = np.array([0.0, 0.0, 0.0])

    camera_translation = np.array([0.4, 0.0, 0]) # 这三个的顺序就是xyz
    camera_rotation = np.array([20.0, 0.0, 0.0]) # 这三个的顺序是pitch在最前面, 后面两个可能是[roll, yaw], 也可能是反过来的

    # 雷达外部参数
    lidar_translation = np.array([0, 0, 0])
    lidar_rotation = np.array([0.0, 0.0, 0.0])

    # 创建相机和雷达的变换矩阵
    camera_matrix = create_transformation_matrix(camera_translation, camera_rotation)
    lidar_matrix = create_transformation_matrix(lidar_translation, lidar_rotation)

    # 计算相机到雷达的相对变换矩阵
    camera_to_lidar_matrix = np.linalg.inv(camera_matrix) @ lidar_matrix

    print("Camera to Lidar Transformation Matrix:")
    print(camera_to_lidar_matrix)

    # extrinsics = np.array(
    #                [[ 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, -0.4],
    #                 [ 0.00000000e+00, 9.39692621e-01, 3.42020143e-01, -3.75877048e-01],
    #                 [ 0.00000000e+00, -3.42020143e-01, 9.39692621e-01, 5.36808057e-01],
    #                 [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]).reshape(4,4)

    extrinsics = camera_to_lidar_matrix
                    
    extrinsics = np.linalg.inv(extrinsics)


    projection(np.array(cloud.points), camera, intrinsics, extrinsics, filter_outliers=False, vis=True)

main()