import os
import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import copy
import torch
from torch.utils.data import Dataset
import open3d as o3d
from natsort import natsorted 
import imageio

from utils.config import SHINEConfig
from utils.tools import get_time
from utils.pose import *
from utils.data_sampler import dataSampler
from utils.semantic_kitti_utils import *
from model.feature_octree import FeatureOctree


# better to write a new dataloader for RGB-D inputs, not always converting them to KITTI Lidar format

class InputDataset(Dataset):
    def __init__(self, config: SHINEConfig, sdf_octree: FeatureOctree = None, color_octree: FeatureOctree = None) -> None:

        super().__init__()

        self.config = config
        self.dtype = config.dtype
        torch.set_default_dtype(self.dtype)
        self.device = config.device

        self.camera2lidar_matrix = self.config.camera_ext_matrix
        self.lidar2camera_matrix = np.linalg.inv(self.camera2lidar_matrix)

        self.calib = {}
        if config.calib_path != '':
            self.calib = read_calib_file(config.calib_path)
        else:
            self.calib['Tr'] = np.eye(4)
        if config.pose_path.endswith('txt'):
            self.poses_w = read_poses_file(config.pose_path, self.calib)
        elif config.pose_path.endswith('csv'):
            self.poses_w = csv_odom_to_transforms(config.pose_path)
        else:
            sys.exit(
            "Wrong pose file format. Please use either *.txt (KITTI format) or *.csv (xyz+quat format)"
            )

        # pose in the reference frame (might be the first frame used)
        self.poses_ref = self.poses_w  # initialize size

        # point cloud files
        self.pc_filenames = natsorted(os.listdir(config.pc_path)) # sort files as 1, 2,… 9, 10 not 1, 10, 100 with natsort
        self.total_pc_count = len(self.pc_filenames) # 文件夹下面总共的点云文件个数

        # image files
        self.image_filenames = natsorted(os.listdir(config.image_path))
        self.total_image_count = len(self.image_filenames)
        if not self.total_image_count == self.total_pc_count:
            sys.exit(
                "Image files count is not equal with point cloud files count."
            )

        # feature octree
        self.sdf_octree = sdf_octree
        self.color_octree = color_octree

        self.last_relative_tran = np.eye(4)

        # initialize the data sampler
        self.sampler = dataSampler(config)
        self.ray_sample_count = config.surface_sample_n + config.free_sample_n

        # merged downsampled point cloud
        # 全局点云地图，只用来记录，最后如果选择要保存点云地图会进行输出
        self.map_down_pc = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox() # 全局点云地图的bbx，这个和下面那个bbx都没作用
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()

        # get the pose in the reference frame
        self.used_pc_count = 0 # 要用的点云文件个数
        begin_flag = False
        self.begin_pose_inv = np.eye(4)
        for frame_id in range(self.total_pc_count):
            if (
                frame_id < config.begin_frame
                or frame_id > config.end_frame
                or frame_id % config.every_frame != 0
            ):
                continue
            if not begin_flag:  # the first frame used
                begin_flag = True
                # 设置参考的坐标，要么参考第一帧，要么参考世界坐标（即使用文件里存的原始坐标）
                # 前者计算第一帧坐标的逆矩阵，后者直接用单位矩阵（下面的global_shift_default还是TO FIX状态，值是0，所以相当于没赋值）
                # 然后用得到的矩阵对每个pose做变换，最后得到poses_ref，也即所谓“参考的坐标”
                if config.first_frame_ref:
                    self.begin_pose_inv = inv(self.poses_w[frame_id])  # T_rw
                else:
                    # just a random number to avoid octree boudnary marching cubes problems on synthetic dataset such as MaiCity(TO FIX)
                    self.begin_pose_inv[2,3] += config.global_shift_default 
            # use the first frame as the reference (identity)
            # 
            self.poses_ref[frame_id] = np.matmul(
                self.begin_pose_inv, self.poses_w[frame_id]
            )
            self.used_pc_count += 1
        # or we directly use the world frame as reference

        # to cope with the gpu memory issue (use cpu memory for the data pool, a bit slower for moving between cpu and gpu)
        # 如果需要用的点云个数大于pc_count_gpu_limit，并且用的是replay不是reg方式，那就把数据存储在cpu memory上，会比在gpu上慢（需要来回切换）
        if self.used_pc_count > config.pc_count_gpu_limit and not config.continual_learning_reg:
            self.pool_device = "cpu"
            self.to_cpu = True
            self.sampler.dev = "cpu"
            print("too many scans, use cpu memory")
        else:
            self.pool_device = config.device
            self.to_cpu = False

        # data pool
        self.coord_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.sdf_label_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.normal_label_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        # self.color_label_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.sem_label_pool = torch.empty((0), device=self.pool_device, dtype=torch.long)
        self.weight_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.sample_depth_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.ray_depth_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.origin_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.time_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.image_pool = None
        self.color_label_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)

    # 读取新的一帧点云和图片并预处理，对这一帧点云进行采样得到采样点，更新octree，更新data pool
    def process_frame(self, frame_id, incremental_on = False):

        pc_radius = self.config.pc_radius
        min_z = self.config.min_z
        max_z = self.config.max_z
        normal_radius_m = self.config.normal_radius_m
        normal_max_nn = self.config.normal_max_nn
        rand_down_r = self.config.rand_down_r
        vox_down_m = self.config.vox_down_m
        sor_nn = self.config.sor_nn
        sor_std = self.config.sor_std

        self.cur_pose_ref = self.poses_ref[frame_id]

        # load point cloud (support *pcd, *ply and kitti *bin format)
        frame_filename = os.path.join(self.config.pc_path, self.pc_filenames[frame_id])
        
        if not self.config.semantic_on:
            # 读取的frame_pc不带有位姿，此时点云坐标在雷达坐标系下面
            frame_pc = self.read_point_cloud(frame_filename)
        else:
            label_filename = os.path.join(self.config.label_path, self.pc_filenames[frame_id].replace('bin','label'))
            frame_pc = self.read_semantic_point_label(frame_filename, label_filename)

        # load image
        image_filename = os.path.join(self.config.image_path, self.image_filenames[frame_id])
        frame_image = imageio.imread(image_filename)

        # block filter: crop the point clouds into a cube
        # 根据设置的bbx对当前读取的一帧点云进行crop。注意这里bbx的中心在坐标系的原点处（z轴不一定，因为z轴max和min值不一样）,
        # 所以这个bbx是对“当前雷达扫描到的一帧点云”的范围进行限制，而不是对全局点云地图做了限制，
        # 并且假如pc_radius的值是默认的20m，那么相当于过滤了手持雷达的人的周围边长20m方形区域外的点
        # 所以如果点云自身带有pose，则点云不再以坐标系原点为中心，但是bbx中心还在原点，那么这个crop过程就出错了，之后的点云需要不带有pose
        bbx_min = np.array([-pc_radius, -pc_radius, min_z])
        bbx_max = np.array([pc_radius, pc_radius, max_z])
        bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
        frame_pc = frame_pc.crop(bbx)

        # surface normal estimation
        if self.config.estimate_normal:
            frame_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius_m, max_nn=normal_max_nn
                )
            )

        # point cloud downsampling
        if self.config.rand_downsample:
            # random downsampling
            frame_pc = frame_pc.random_down_sample(sampling_ratio=rand_down_r)
        else:
            # voxel downsampling
            frame_pc = frame_pc.voxel_down_sample(voxel_size=vox_down_m)

        # apply filter (optional)
        if self.config.filter_noise:
            frame_pc = frame_pc.remove_statistical_outlier(
                sor_nn, sor_std, print_progress=False
            )[0]

        # load the label from the color channel of frame_pc
        if self.config.semantic_on:
            frame_sem_label = np.asarray(frame_pc.colors)[:,0]*255.0 # from [0-1] tp [0-255]
            frame_sem_label = np.round(frame_sem_label, 0) # to integer value
            sem_label_list = list(frame_sem_label)
            frame_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in sem_label_list]
            frame_sem_rgb = np.asarray(frame_sem_rgb, dtype=np.float64)/255.0
            frame_pc.colors = o3d.utility.Vector3dVector(frame_sem_rgb)
        
        # 去掉不能映射到相机图片中的点
        # 将点从雷达坐标系转到相机坐标系
        frame_pc_points = np.asarray(frame_pc.points, dtype=np.float64)
        points3d_lidar = np.asarray(frame_pc.points, dtype=np.float64)
        # points3d_lidar = frame_pc.clone()
        points3d_lidar = np.insert(points3d_lidar, 3, 1, axis=1)
        points3d_camera = self.lidar2camera_matrix @ points3d_lidar.T
        H, W, fx, fy, cx, cy, = self.config.H, self.config.W, self.config.fx, self.config.fy, self.config.cx, self.config.cy
        K = np.array([[fx, .0, cx, .0], [.0, fy, cy, .0], [.0, .0, 1.0, .0]]).reshape(3, 4)
        # 过滤掉相机坐标系内位于相机之后的点
        tmp_mask = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, tmp_mask]
        frame_pc_points = frame_pc_points[tmp_mask]
        # 从相机坐标系映射到uv平面坐标
        points2d_camera = K @ points3d_camera
        points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 操作之后points2d_camera维度:[n, 2]
        # 过滤掉uv平面坐标内在图像外的点
        # TODO points2d_camera[:, 1]是否真的对应H? 以及下面的frame_image[points2d_camera[:,1].astype(int), points2d_camera[:,0]行和列是否正确?
        tmp_mask = np.logical_and(
            (points2d_camera[:, 1] < H) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < W) & (points2d_camera[:, 0] > 0)
        )
        points2d_camera = points2d_camera[tmp_mask]
        # points3d_camera = (points3d_camera.T)[tmp_mask] # 操作之后points3d_camera维度: [n, 4]
        frame_pc_points = frame_pc_points[tmp_mask]
        # 取出图像内uv坐标对应的颜色
        frame_color_label = torch.tensor(frame_image[points2d_camera[:,1].astype(int), points2d_camera[:,0].astype(int)], 
                                         device=self.pool_device)
        frame_pc.points = o3d.utility.Vector3dVector(frame_pc_points)
        # 上面这几步旨在将frame_pc中的点映射到图片上，对应位置的像素点颜色形成一个新的pool，顺序与frame_pc中点的顺序相同
        # 之后get_batch的时候，类似于从ray_depth_pool获取ray的深度，同样可以直接获取ray的颜色
        

        # points3d_lidar = np.asarray(frame_pc.points, dtype=np.float64)
        # # points3d_lidar = frame_pc.clone()
        # points3d_lidar = np.insert(points3d_lidar, 3, 1, axis=1)
        # points3d_camera = self.lidar2camera_matrix @ points3d_lidar.T
        # H, W, fx, fy, cx, cy, = self.config.H, self.config.W, self.config.fx, self.config.fy, self.config.cx, self.config.cy
        # K = np.array([[fx, .0, cx, .0], [.0, fy, cy, .0], [.0, .0, 1.0, .0]]).reshape(3, 4)
        # # 过滤掉相机坐标系内位于相机之后的点
        # mask_one = points3d_camera[2, :] > 0.0 #mask_one是n维的
        # # 从相机坐标系映射到uv平面坐标
        # points2d_camera = K @ points3d_camera
        # points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 操作之后points2d_camera维度:[n, 2]
        # # 过滤掉uv平面坐标内在图像外的点
        # mask_two = np.logical_and(
        #     (points2d_camera[:, 1] < H) & (points2d_camera[:, 1] > 0),
        #     (points2d_camera[:, 0] < W) & (points2d_camera[:, 0] > 0)
        # ) # mask_two也是n维的
        # mask = mask_one & mask_two # 最后mask是n维的，也就是和points2d_camera中点的个数一样
        # # 取出图像内uv坐标对应的颜色
        # frame_color_label = torch.zeros(points2d_camera.shape[0], 3, device=self.pool_device, dtype=torch.uint8)
        # frame_color_label[mask] = torch.tensor(frame_image[points2d_camera[mask,1].astype(int), points2d_camera[mask,0].astype(int)],
        #                                        device=self.pool_device)
        # frame_color_label[~mask] = torch.tensor([255, 255, 255], device=self.pool_device, dtype=torch.uint8)


        # 乘以scale归一化到[-1,1]区间，只需要对平移部分乘就行
        frame_origin = self.cur_pose_ref[:3, 3] * self.config.scale  # translation part
        frame_origin_torch = torch.tensor(frame_origin, dtype=self.dtype, device=self.pool_device)

        cur_camera_pose_ref = self.lidar2camera_matrix @ self.cur_pose_ref
        camera_origin = cur_camera_pose_ref[:3, 3] * self.config.scale
        camera_origin_torch = torch.tensor(camera_origin, dtype=self.dtype, device=self.pool_device)

        # transform to reference frame 
        frame_pc = frame_pc.transform(self.cur_pose_ref)
        # make a backup for merging into the map point cloud
        frame_pc_clone = copy.deepcopy(frame_pc)
        frame_pc_clone = frame_pc_clone.voxel_down_sample(voxel_size=self.config.map_vox_down_m) # for smaller memory cost
        self.map_down_pc += frame_pc_clone # 把transform后的点云和全局地图合并在一起
        self.cur_frame_pc = frame_pc_clone # sine_incre里面用这个update vis

        self.map_bbx = self.map_down_pc.get_axis_aligned_bounding_box()
        self.cur_bbx = self.cur_frame_pc.get_axis_aligned_bounding_box()
        # and scale to [-1,1] coordinate system
        frame_pc_s = frame_pc.scale(self.config.scale, center=(0,0,0))

        frame_pc_s_torch = torch.tensor(np.asarray(frame_pc_s.points), dtype=self.dtype, device=self.pool_device)

        frame_normal_torch = None
        if self.config.estimate_normal:
            frame_normal_torch = torch.tensor(np.asarray(frame_pc_s.normals), dtype=self.dtype, device=self.pool_device)

        frame_label_torch = None
        if self.config.semantic_on:
            frame_label_torch = torch.tensor(frame_sem_label, dtype=self.dtype, device=self.pool_device)

        # print("Frame point cloud count:", frame_pc_s_torch.shape[0])

        # 经过transform(self.cur_pose_ref)之后点云坐标从雷达坐标系移动到世界坐标系，scale之后就是[-1,1]范围内的世界坐标系下的点云
        # sample函数内frame_pc_s_torch会根据frame_origin_torch平移，在scale后的世界坐标系下移动到以雷达中心为参考点
        # 为了sample里面计算distances的时候计算的是到相机的距离而不是到雷达的距离
        # 传递的第二个参数改成camera_origin_torch，计算自相机pose的平移部分，这样sample内的点云平移时就会移动到以scale后的相机为参考点
        # 采样的点就会在点云点和相机的连线上

        # sampling the points
        # coord: 所有采样点坐标(scale后的); sdf_label: 所有采样点sdf真值(有正负之分)(scale后的); weight: 所有采样点的类型正的是surface，负的是free; 
        # sample_depth: 所有采样点的真实深度(不是scale后的); ray_depth: 当前帧点云中所有点的真实深度
        (coord, sdf_label, normal_label, sem_label, weight, sample_depth, ray_depth) = \
            self.sampler.sample(frame_pc_s_torch, camera_origin_torch, \
            frame_normal_torch, frame_label_torch)
        
        origin_repeat = frame_origin_torch.repeat(coord.shape[0], 1)
        time_repeat = torch.tensor(frame_id, dtype=self.dtype, device=self.pool_device).repeat(coord.shape[0])

        # update feature octree
        if self.sdf_octree is not None:
            if self.config.octree_from_surface_samples:
                # 如果设置了octree_from_surface_samples，也就是用surface类型采样点来更新octree，就传递surface类型的采样点到update里，
                # 并且注意这里的coord是scale后的坐标
                # update with the sampled surface points
                self.sdf_octree.update(coord[weight > 0, :].to(self.device), incremental_on)
            else:
                # 否则使用当前帧点云来更新octree，区别在于使用采样点更新的时候memory占用显然会更大，但是更robust
                # 并且这里的点云坐标也是scale后的坐标
                # update with the original points
                self.sdf_octree.update(frame_pc_s_torch.to(self.device), incremental_on)

        if self.color_octree is not None:
            if self.config.color_octree_from_surface_samples:
                self.color_octree.update(coord[weight > 0, :].to(self.device), incremental_on)
            else:
                self.color_octree.update(frame_pc_s_torch.to(self.device), incremental_on)  

        # get the data pool ready for training
        # TODO camera_origin_torch用不用保存，以及和origin_pool之间关系
        # ray-wise samples order
        if incremental_on: # for the incremental mapping with feature update regularization
            self.coord_pool = coord
            self.sdf_label_pool = sdf_label
            self.normal_label_pool = normal_label
            self.sem_label_pool = sem_label
            # self.color_label_pool = color_label
            self.weight_pool = weight
            self.sample_depth_pool = sample_depth
            self.ray_depth_pool = ray_depth
            self.origin_pool = origin_repeat
            self.time_pool = time_repeat
            self.color_label_pool = frame_color_label
            # 两种提供rgb真值的方案：一种是把所有点云点要映射到的图片像素点存起来，get batch的时候查询；一种是get_batch的时候现场计算随机到的点云点会映射到哪些像素点
        
        else: # batch processing    
            # using a sliding window for the data pool
            if self.config.window_replay_on: 
                pool_relative_dist = (self.coord_pool - frame_origin_torch).norm(2, dim=-1)
                filter_mask = pool_relative_dist < self.config.window_radius * self.config.scale

                # and also have two filter mask options (delta frame, distance)
                # print(filter_mask)

                self.coord_pool = self.coord_pool[filter_mask]
                self.weight_pool = self.weight_pool[filter_mask]

                # FIX ME for ray-wise sampling
                # self.sample_depth_pool = self.sample_depth_pool[filter_mask]
                # self.ray_depth_pool = self.ray_depth_pool[filter_mask]
                
                self.sdf_label_pool = self.sdf_label_pool[filter_mask]
                self.origin_pool = self.origin_pool[filter_mask]
                self.time_pool = self.time_pool[filter_mask]
                
                if normal_label is not None:
                    self.normal_label_pool = self.normal_label_pool[filter_mask]
                if sem_label is not None:
                    self.sem_label_pool = self.sem_label_pool[filter_mask]
            
            # or we will simply use all the previous samples

            # concat with current observations
            self.coord_pool = torch.cat((self.coord_pool, coord.to(self.pool_device)), 0)            
            self.weight_pool = torch.cat((self.weight_pool, weight.to(self.pool_device)), 0)
            if self.config.ray_loss:
                self.sample_depth_pool = torch.cat((self.sample_depth_pool, sample_depth.to(self.pool_device)), 0)
                self.ray_depth_pool = torch.cat((self.ray_depth_pool, ray_depth.to(self.pool_device)), 0)
            else:
                self.sdf_label_pool = torch.cat((self.sdf_label_pool, sdf_label.to(self.pool_device)), 0)
                self.origin_pool = torch.cat((self.origin_pool, origin_repeat.to(self.pool_device)), 0)
                self.time_pool = torch.cat((self.time_pool, time_repeat.to(self.pool_device)), 0)

            if normal_label is not None:
                self.normal_label_pool = torch.cat((self.normal_label_pool, normal_label.to(self.pool_device)), 0)
            else:
                self.normal_label_pool = None
            
            if sem_label is not None:
                self.sem_label_pool = torch.cat((self.sem_label_pool, sem_label.to(self.pool_device)), 0)
            else:
                self.sem_label_pool = None

    def read_point_cloud(self, filename: str):
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
        preprocessed_points = self.preprocess_kitti(
            points, self.config.min_z, self.config.min_range
        )
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(preprocessed_points) # Vector3dVector is faster for np.float64 
        return pc_out

    def read_semantic_point_label(self, bin_filename: str, label_filename: str):

        # read point cloud (kitti *.bin format)
        if ".bin" in bin_filename:
            points = np.fromfile(bin_filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        else:
            sys.exit(
                "The format of the imported point cloud is wrong (support only *bin)"
            )

        # read point cloud labels (*.label format)
        if ".label" in label_filename:
            labels = np.fromfile(label_filename, dtype=np.uint32).reshape((-1))
        else:
            sys.exit(
                "The format of the imported point labels is wrong (support only *label)"
            )

        points, sem_labels = self.preprocess_sem_kitti(
            points, labels, self.config.min_z, self.config.min_range, filter_moving=self.config.filter_moving_object
        )

        sem_labels = (np.asarray(sem_labels, dtype=np.float64)/255.0).reshape((-1, 1)).repeat(3, axis=1) # label 

        # TODO: better to use o3d.t.geometry.PointCloud(device)
        # a bit too cubersome
        # then you can use sdf_map_pc.point['positions'], sdf_map_pc.point['intensities'], sdf_map_pc.point['labels']
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(points) # Vector3dVector is faster for np.float64 
        pc_out.colors = o3d.utility.Vector3dVector(sem_labels)

        return pc_out

    def preprocess_kitti(self, points, z_th=-3.0, min_range=2.5):
        # filter the outliers
        # 去掉z轴值小于z_th的点，以及和原点间距离小于min_range的点
        z = points[:, 2]
        points = points[z > z_th]
        points = points[np.linalg.norm(points, axis=1) >= min_range]
        return points

    def preprocess_sem_kitti(self, points, labels, min_range=2.75, filter_outlier = True, filter_moving = True):
        # TODO: speed up
        sem_labels = np.array(labels & 0xFFFF)

        range_filtered_idx = np.linalg.norm(points, axis=1) >= min_range
        points = points[range_filtered_idx]
        sem_labels = sem_labels[range_filtered_idx]

        # filter the outliers according to semantic labels
        if filter_moving:
            filtered_idx = sem_labels < 100
            points = points[filtered_idx]
            sem_labels = sem_labels[filtered_idx]

        if filter_outlier:
            filtered_idx = (sem_labels != 1) # not outlier
            points = points[filtered_idx]
            sem_labels = sem_labels[filtered_idx]
        
        sem_labels_main_class = np.array([sem_kitti_learning_map[sem_label] for sem_label in sem_labels]) # get the reduced label [0-20]

        return points, sem_labels_main_class
    
    def write_merged_pc(self, out_path):
        map_down_pc_out = copy.deepcopy(self.map_down_pc)
        map_down_pc_out.transform(inv(self.begin_pose_inv)) # back to world coordinate (if taking the first frame as reference)
        o3d.io.write_point_cloud(out_path, map_down_pc_out) 
        print("save the merged point cloud map to %s\n" % (out_path))    

    def __len__(self) -> int:
        if self.config.ray_loss:
            return self.ray_depth_pool.shape[0]  # ray count
        else:
            return self.sdf_label_pool.shape[0]  # point sample count

    # deprecated
    def __getitem__(self, index: int):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            sample_index = torch.range(0, self.ray_sample_count - 1, dtype=int)
            sample_index += index * self.ray_sample_count

            coord = self.coord_pool[sample_index, :]
            # sdf_label = self.sdf_label_pool[sample_index]
            # normal_label = self.normal_label_pool[sample_index]
            # sem_label = self.sem_label_pool[sample_index]
            sample_depth = self.sample_depth_pool[sample_index]
            ray_depth = self.ray_depth_pool[index]

            return coord, sample_depth, ray_depth

        else:  # use point sample
            coord = self.coord_pool[index, :]
            sdf_label = self.sdf_label_pool[index]
            # normal_label = self.normal_label_pool[index]
            # sem_label = self.sem_label_pool[index]
            weight = self.weight_pool[index]

            return coord, sdf_label, weight
    
    def get_batch(self):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            train_ray_count = self.ray_depth_pool.shape[0]
            ray_index = torch.randint(0, train_ray_count, (self.config.bs,), device=self.pool_device)

            ray_index_repeat = (ray_index * self.ray_sample_count).repeat(self.ray_sample_count, 1)
            sample_index = ray_index_repeat + torch.arange(0, self.ray_sample_count,\
                 dtype=int, device=self.device).reshape(-1, 1)
            index = sample_index.transpose(0,1).reshape(-1)

            # ray_loss控制使用depth估计和可微分渲染，都需要ray sample，也就是按照ray进行采样，
            # 也即把这一帧点云里每个点看成一个ray，由于前面process_frame的时候已经在每个点云点的ray上采样了六个sample点，
            # 所以ray sample的每次采样都要把这六个点都包含进去
            # 没看懂上面过程的话可以设个值试一下
            # 上面的ray_index是采样了4096(假设bs是4096)个点云点，假设ray_sample_count是6(surface 3+ free 3)，
            # 那么这里的coord的维度显然是(4096*6, 3)
            coord = self.coord_pool[index, :].to(self.device)
            weight = self.weight_pool[index].to(self.device)
            sample_depth = self.sample_depth_pool[index].to(self.device)

            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :].to(self.device)
            else: 
                normal_label = None

            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[ray_index * self.ray_sample_count].to(self.device) # one semantic label for one ray
            else: 
                sem_label = None

            ray_depth = self.ray_depth_pool[ray_index].to(self.device)
            color_label = self.color_label_pool[ray_index].to(self.device)

            return coord, sample_depth, ray_depth, normal_label, sem_label, weight, color_label
        
        else: # use point sample
            # 在不用ray_loss的情况下，只使用sdf loss，那计算loss的方法就不再以ray为单位，而是以sample point为单位，
            # 所以直接随机batch size个范围是[0, sample points个数)的index就行
            # 也即采样单位不是ray，而是sample points，因为每个sample point都能单独计算sdf loss，而深度估计和可微渲染都需要一条ray上完整六个采样点才能算loss
            train_sample_count = self.sdf_label_pool.shape[0]
            index = torch.randint(0, train_sample_count, (self.config.bs,), device=self.pool_device)
            coord = self.coord_pool[index, :].to(self.device)
            sdf_label = self.sdf_label_pool[index].to(self.device)
            origin = self.origin_pool[index].to(self.device)
            ts = self.time_pool[index].to(self.device) # frame number or the timestamp

            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :].to(self.device)
            else: 
                normal_label = None
            
            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[index].to(self.device)
            else: 
                sem_label = None

            weight = self.weight_pool[index].to(self.device)

            return coord, sdf_label, origin, ts, normal_label, sem_label, weight


    def get_batch_all(self):
        # use ray sample (each sample containing all the sample points on the ray)
        train_ray_count = self.ray_depth_pool.shape[0]
        ray_index = torch.randint(0, train_ray_count, (self.config.bs,), device=self.pool_device)

        ray_index_repeat = (ray_index * self.ray_sample_count).repeat(self.ray_sample_count, 1)
        sample_index = ray_index_repeat + torch.arange(0, self.ray_sample_count,\
                dtype=int, device=self.device).reshape(-1, 1)
        index = sample_index.transpose(0,1).reshape(-1)

        # ray_loss控制使用depth估计和可微分渲染，都需要ray sample，也就是按照ray进行采样，
        # 也即把这一帧点云里每个点看成一个ray，由于前面process_frame的时候已经在每个点云点的ray上采样了六个sample点，
        # 所以ray sample的每次采样都要把这六个点都包含进去
        # 没看懂上面过程的话可以设个值试一下
        # 上面的ray_index是采样了4096(假设bs是4096)个点云点，假设ray_sample_count是6(surface 3+ free 3)，
        # 那么这里的coord的维度显然是(4096*6, 3)
        coord = self.coord_pool[index, :].to(self.device)
        sdf_label = self.sdf_label_pool[index].to(self.device)
        weight = self.weight_pool[index].to(self.device)
        sample_depth = self.sample_depth_pool[index].to(self.device)

        if self.normal_label_pool is not None:
            normal_label = self.normal_label_pool[index, :].to(self.device)
        else: 
            normal_label = None

        if self.sem_label_pool is not None:
            sem_label = self.sem_label_pool[ray_index * self.ray_sample_count].to(self.device) # one semantic label for one ray
        else: 
            sem_label = None

        ray_depth = self.ray_depth_pool[ray_index].to(self.device)
        color_label = self.color_label_pool[ray_index].to(self.device)

        return coord, sample_depth, ray_depth, normal_label, sem_label, weight, color_label, sdf_label

    def filter_pc(self, frame_pc):
        """
        将新帧点云映射到图像uv坐标系内, 并删除映射后在图像外的点

        Args:
            frame_pc: 新一帧点云，不带位姿，因此是雷达坐标系下的

        Returns:
            将映射到uv坐标系后位于图像外的点删除后的点云
        """
        # 得到grid中每个点的坐标
        H, W, fx, fy, cx, cy, = self.config.H, self.config.W, self.config.fx, self.config.fy, self.config.cx, self.config.cy

        points = frame_pc.clone()
        # 输入的frame_pc是不带有位姿的，所以省略了去掉位姿即移动回相机坐标系的步骤
        # points_bak = frame_pc.clone()
        # c2w = c2w.cpu().numpy()
        # w2c = np.linalg.inv(c2w)
        # ones = np.ones_like(frame_pc[:, 0]).reshape(-1, 1)
        # homo_vertices = np.concatenate(
        #     [frame_pc, ones], axis=1).reshape(-1, 4, 1)
        # cam_cord_homo = w2c@homo_vertices
        # cam_cord = cam_cord_homo[:, :3]

        points = self.lidar2camera_matrix * points
        
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        # points[:, 0] *= -1
        uv = K@points
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        # 利用remap函数得到深度图中每个上面得到的uv坐标处的深度值
        # remap_chunk = int(3e4)
        # depths = []
        # for i in range(0, uv.shape[0], remap_chunk):
        #     depths += [cv2.remap(depth_np,
        #                          uv[i:i+remap_chunk, 0],
        #                          uv[i:i+remap_chunk, 1],
        #                          interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        # depths = np.concatenate(depths, axis=0)

        # 第一个筛选条件：筛选变到uv坐标后仍在相机H W范围内的grid点
        edge = 0
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        mask = mask.reshape(-1)

        frame_pc = frame_pc[mask]
        return frame_pc

