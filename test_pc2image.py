import open3d as o3d
import numpy as np
from copy import deepcopy as dc
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from scipy.spatial.transform import Rotation as R


def projection(points, camera_scan, intrinsics, extrinsics, filter_outliers=True, vis=True):
    trans_lidar_to_camera = extrinsics

    points3d_lidar = points
    points3d_lidar = np.insert(points3d_lidar,3,1,axis=1)
    points3d_lidar = points3d_lidar @ np.array([0, 0, -1, 0, 
                                                1, 0, 0, 0, 
                                                0, -1, 0, 0, 
                                                0, 0, 0, 1]).reshape(4,4)

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
        colors = cm.jet(distances / np.max(distances))
        plt.gca().scatter(points2d_camera[:, 0], points2d_camera[:, 1], color=colors, s=1)
        plt.show()
def main():
    cloud = o3d.io.read_point_cloud("/home/wuchenyang/NeRF/dataset/carla/2023_11_14/livox_pc/157.555801243.pcd")
    cloud.voxel_down_sample(4)
    camera = mpimg.imread("/home/wuchenyang/NeRF/dataset/carla/2023_11_14/image/157.555801.png")
    intrinsics = np.array( ### edit
        [[400., 0.0, 400., 0],
        [0.0, 400., 300., 0],
        [0.0, 0.0, 1.0, 0]])
    extrinsics = np.array(### edit
        [ 0.93969262, 0., -0.34202014, 0.,
                      0.,  1., 0., -0.4,
                      0.34202014,  0.,  0.93969262, 0.,
                      0., 0., 0., 1.]).reshape(4,4)
    extrinsics = np.linalg.inv(extrinsics)


    projection(np.array(cloud.points), camera, intrinsics, extrinsics, filter_outliers=False, vis=True)

main()