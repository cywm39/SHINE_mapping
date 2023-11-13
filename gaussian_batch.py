import torch
import os
import math
import random
import numpy as np
import torchvision
import open3d as o3d
import getpass
import wandb
import shutil
import time
from datetime import datetime
from natsort import natsorted 
from utils.pose import read_poses_file
from PIL import Image
import collections
import json
import torch.nn as nn
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils import ms
from utils.gaussian_config import Gaussian_config

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(10 * torch.ones(1, device='cuda'))

class GsDataset:
    def __init__(self, device, resolution=(256,256), pose_file_path='', image_folder_path=''):
        def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
            Rt = np.zeros((4, 4))
            Rt[:3, :3] = R.transpose()
            Rt[:3, 3] = t
            Rt[3, 3] = 1.0
            C2W = np.linalg.inv(Rt)
            cam_center = C2W[:3, 3]
            cam_center = (cam_center + translate) * scale
            C2W[:3, 3] = cam_center
            Rt = np.linalg.inv(C2W)
            return np.float32(Rt)

        def load_image_camera_from_transforms(device, resolution, white_background=False, pose_file_path='', image_folder_path=''):
            class Camera:
                def __init__(self, device, uid, image_data, image_path, image_name, image_width, image_height, R, t, FovX, FovY, 
                             znear=0.01, zfar=100.0, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
                    def getProjectionMatrix(znear, zfar, fovX, fovY):
                        tanHalfFovY = math.tan((fovY / 2))
                        tanHalfFovX = math.tan((fovX / 2))
                        top = tanHalfFovY * znear
                        bottom = -top
                        right = tanHalfFovX * znear
                        left = -right
                        P = torch.zeros(4, 4)
                        z_sign = 1.0
                        P[0, 0] = 2.0 * znear / (right - left)
                        P[1, 1] = 2.0 * znear / (top - bottom)
                        P[0, 2] = (right + left) / (right - left)
                        P[1, 2] = (top + bottom) / (top - bottom)
                        P[3, 2] = z_sign
                        P[2, 2] = z_sign * zfar / (zfar - znear)
                        P[2, 3] = -(zfar * znear) / (zfar - znear)
                        return P

                    # 每帧camera的各种参数
                    self.uid = uid
                    image_data = torch.from_numpy(np.array(image_data)) / 255.0
                    self.image_goal = image_data.clone().clamp(0.0, 1.0).permute(2, 0, 1) #.to(device)
                    self.image_tidy = image_data.permute(2, 0, 1) if len(image_data.shape) == 3 else image_data.unsqueeze(dim=-1).permute(2, 0, 1)
                    self.image_path = image_path
                    self.image_name = image_name
                    self.image_width = image_width
                    self.image_height = image_height
                    self.R = R  # w2c中R transpose过的结果
                    self.t = t  # w2c中的t
                    self.FovX = FovX
                    self.FovY = FovY
                    self.znear = znear # default
                    self.zfar = zfar # default
                    self.trans = trans # default
                    self.scale = scale # default
                    self.world_view_transform = torch.tensor(getWorld2View2(R, t, self.trans, self.scale)).transpose(0, 1).to(device) # w2c的转置
                    self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FovX, 
                                                                 fovY=self.FovY).transpose(0,1).to(device) # 三维到二维的投影矩阵，似乎也能从二维恢复三维点信息
                    # w2c乘以三维到二维的投影矩阵，得到的结果可以直接让一个世界坐标系下的三维点投影到二维上
                    self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                    self.camera_center = self.world_view_transform.inverse()[3, :3]

            def fov2focal(fov, pixels):
                return pixels / (2 * math.tan(fov / 2))
            def focal2fov(focal, pixels):
                return 2*math.atan(pixels/(2*focal))

            image_camera = []

            calib = {}
            calib['Tr'] = np.eye(4)
            poses = read_poses_file(pose_file_path, calib)
            image_filenames = natsorted(os.listdir(image_folder_path))
            pose_index = 0

            fovx = 1.275735492592156

            for filename in image_filenames:
                image_path = os.path.join(image_folder_path, filename)
                image_norm = np.array(Image.open(image_path).convert("RGBA")) / 255.0
                image_back = np.array((np.array([1.,1.,1.]) if white_background else np.array([0., 0., 0.])) * (1. - image_norm[:, :, 3:4]) * 255, 
                                        dtype=np.byte)
                image_fore = np.array(image_norm[:,:,:3] * image_norm[:, :, 3:4] * 255, dtype=np.byte)
                image_data = Image.fromarray(image_fore + image_back, "RGB")
                c2w = poses[pose_index] #NeRF 'transform_matrix' is a camera-to-world transform
                # c2w[:3, 1:3] *= -1  #change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                w2c = np.linalg.inv(c2w)  #get the world-to-camera transform and set R, T
                R,t = np.transpose(w2c[:3,:3]), w2c[:3, 3]  # R is stored transposed due to 'glm' in CUDA code
                fovy = focal2fov(fov2focal(fovx, image_data.size[0]), image_data.size[1])
                # print('fovy: ' + str(fovy))
                camera = Camera(device=device, uid=pose_index, image_data=image_data, image_path=image_path, image_name=os.path.basename(image_path), 
                                image_width=image_data.size[0], image_height=image_data.size[1], R=R, t=t, FovX=fovx, FovY=fovy)
                image_camera.append(camera)
                pose_index += 1

            # with open(os.path.join(path, transforms_file)) as json_file:
            #     transforms_json = json.load(json_file)
            #     fovx = transforms_json["camera_angle_x"]
            #     for idx, frame in enumerate(transforms_json["frames"]): 
            #         image_path = os.path.join(path, frame["file_path"])
            #         image_norm = np.array(Image.open(image_path).convert("RGBA")) / 255.0
            #         image_back = np.array((np.array([1.,1.,1.]) if white_background else np.array([0., 0., 0.])) * (1. - image_norm[:, :, 3:4]) * 255, 
            #                               dtype=np.byte)
            #         image_fore = np.array(image_norm[:,:,:3] * image_norm[:, :, 3:4] * 255, dtype=np.byte)
            #         image_data = Image.fromarray(image_fore + image_back, "RGB").resize(resolution) 
            #         c2w = np.array(frame["transform_matrix"])  #NeRF 'transform_matrix' is a camera-to-world transform
            #         c2w[:3, 1:3] *= -1  #change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            #         w2c = np.linalg.inv(c2w)  #get the world-to-camera transform and set R, T
            #         R,t = np.transpose(w2c[:3,:3]), w2c[:3, 3]  # R is stored transposed due to 'glm' in CUDA code
            #         fovy = focal2fov(fov2focal(fovx, image_data.size[0]), image_data.size[1])
            #         camera = Camera(device=device, uid=idx, image_data=image_data, image_path=image_path, image_name=os.path.basename(image_path), 
            #                         image_width=image_data.size[0], image_height=image_data.size[1], R=R, t=t, FovX=fovx, FovY=fovy)
            #         image_camera.append(camera)
            return image_camera

        def getNerfppNorm(cam_info):
            def get_center_and_diag(cam_centers):
                cam_centers = np.hstack(cam_centers)
                avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
                center = avg_cam_center
                dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
                diagonal = np.max(dist)
                return center.flatten(), diagonal
            cam_centers = []
            for cam in cam_info:
                W2C = getWorld2View2(cam.R, cam.t)
                C2W = np.linalg.inv(W2C)
                cam_centers.append(C2W[:3, 3:4])
            center, diagonal = get_center_and_diag(cam_centers)
            radius = diagonal * 1.1
            translate = -center
            return {"translate": translate, "radius": radius}

        # 读取每一帧的各种相机参数，返回一个list
        self.image_camera = load_image_camera_from_transforms(device, resolution, 
                                                              pose_file_path=pose_file_path, image_folder_path=image_folder_path)
        # radius的定义：对所有帧的相机中心点坐标求平均值得到相机路径的几何中心，所有帧的相机中心点坐标和几何中心距离最远的那个距离*1.1就是radius
        self.cameras_extent = getNerfppNorm(self.image_camera)["radius"]

class GsNetwork:
    def __init__(self, device, percent_dense=0.01, max_sh_degree=3, point_number=100000, pc_path = ''):
        self.percent_dense = percent_dense
        self.max_sh_degree, self.now_sh_degree = max_sh_degree, 0  #spherical-harmonics

        pc_load = o3d.io.read_point_cloud(pc_path)
        pc_points = np.asarray(pc_load.points, dtype=np.float64)
        point_number = pc_points.shape[0]
        points = torch.tensor(pc_points, device=device).float()
        
        # [-1.3, 1.3]范围
        # points = torch.rand(point_number, 3).float().to(device) * 2.6 - 1.3  #normals=torch.zeros(point_number, 3)
        features = torch.cat((torch.rand(point_number, 3, 1).float().to(device) / 5.0 + 0.4, 
                              torch.zeros((point_number, 3, (self.max_sh_degree + 1) ** 2 -1)).float().to(device)), dim=-1)
        scale = torch.log(torch.sqrt(torch.clamp_min(distCUDA2(points).float(), 0.0000001)))[...,None].repeat(1, 3)
        rotation = torch.cat((torch.ones((point_number, 1)).float().to(device), 
                              torch.zeros((point_number, 3)).float().to(device)), dim=1) 
        opacity = torch.log((torch.ones((point_number, 1)).float().to(device) * 0.1) 
                            / (1. - (torch.ones((point_number, 1)).float().to(device) * 0.1)))
        # 下面是每个点的四种待优化变量
        # 位置
        self._xyz = nn.Parameter(points.requires_grad_(True))
        # 这两个是球谐函数
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # 缩放和旋转组成协方差矩阵
        self._scaling = nn.Parameter(scale.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        # 不透明度
        self._opacity = nn.Parameter(opacity.requires_grad_(True))

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            def build_scaling_rotation(s, r):
                def build_rotation(r):
                    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
                    q = r / norm[:, None]
                    R = torch.zeros((q.size(0), 3, 3), device='cuda')
                    r = q[:, 0]
                    x = q[:, 1]
                    y = q[:, 2]
                    z = q[:, 3]
                    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
                    R[:, 0, 1] = 2 * (x*y - r*z)
                    R[:, 0, 2] = 2 * (x*z + r*y)
                    R[:, 1, 0] = 2 * (x*y + r*z)
                    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
                    R[:, 1, 2] = 2 * (y*z - r*x)
                    R[:, 2, 0] = 2 * (x*z - r*y)
                    R[:, 2, 1] = 2 * (y*z + r*x)
                    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
                    return R
                L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
                R = build_rotation(r)
                L[:,0,0] = s[:,0]
                L[:,1,1] = s[:,1]
                L[:,2,2] = s[:,2]
                L = R @ L
                return L

            def strip_symmetric(sym):
                def strip_lowerdiag(L):
                    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
                    uncertainty[:, 0] = L[:, 0, 0]
                    uncertainty[:, 1] = L[:, 0, 1]
                    uncertainty[:, 2] = L[:, 0, 2]
                    uncertainty[:, 3] = L[:, 1, 1]
                    uncertainty[:, 4] = L[:, 1, 2]
                    uncertainty[:, 5] = L[:, 2, 2]
                    return uncertainty
                return strip_lowerdiag(sym)

            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm 
      
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = lambda x: torch.log(x/(1.-x))   #inverse-sigmoid

        self.max_radii2D = torch.zeros((point_number)).float().to(device)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

    def move2cpu(self):
        self._xyz = self._xyz.to("cpu")
        self._features_dc = self._features_dc.to('cpu')
        self._features_rest = self._features_rest.to('cpu')
        self._scaling = self._scaling.to('cpu')
        self._rotation = self._rotation.to('cpu')
        self._opacity = self._opacity.to('cpu')
        self.max_radii2D = self.max_radii2D.to('cpu')
        self.xyz_gradient_accum = self.xyz_gradient_accum.to('cpu')
        self.denom = self.denom.to('cpu')

    def move2cuda(self):
        self._xyz = self._xyz.to("cuda")
        self._features_dc = self._features_dc.to('cuda')
        self._features_rest = self._features_rest.to('cuda')
        self._scaling = self._scaling.to('cuda')
        self._rotation = self._rotation.to('cuda')
        self._opacity = self._opacity.to('cuda')
        self.max_radii2D = self.max_radii2D.to('cuda')
        self.xyz_gradient_accum = self.xyz_gradient_accum.to('cuda')
        self.denom = self.denom.to('cuda')

    def oneupSHdegree(self):
        if self.now_sh_degree < self.max_sh_degree:
            self.now_sh_degree += 1

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, optimizer):
        def densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer):
            def cat_tensors_to_optimizer(tensors_dict, optimizer):
                optimizable_tensors = {}
                for group in optimizer.param_groups:
                    assert len(group["params"]) == 1
                    extension_tensor = tensors_dict[group["name"]]
                    stored_state = optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                        del optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                        optimizer.state[group['params'][0]] = stored_state
                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                        optimizable_tensors[group["name"]] = group["params"][0]
                return optimizable_tensors

            d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest, "opacity": new_opacities, "scaling" : new_scaling, "rotation" : new_rotation}
            optimizable_tensors = cat_tensors_to_optimizer(d, optimizer)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        def prune_points(mask, optimizer):
            def _prune_optimizer(mask, optimizer):
                optimizable_tensors = {}
                for group in optimizer.param_groups:
                    stored_state = optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                        del optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                        optimizer.state[group['params'][0]] = stored_state
                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                        optimizable_tensors[group["name"]] = group["params"][0]
                return optimizable_tensors

            valid_points_mask = ~mask
            optimizable_tensors = _prune_optimizer(valid_points_mask, optimizer)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]

        def densify_and_clone(grads, grad_threshold, scene_extent, optimizer):            
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)   #extract points that satisfy the gradient condition         
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer)

        def densify_and_split(grads, grad_threshold, scene_extent, optimizer, N=2):
            def build_rotation(r):
                norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
                q = r / norm[:, None]
                R = torch.zeros((q.size(0), 3, 3), device=r.device)
                r = q[:, 0]
                x = q[:, 1]
                y = q[:, 2]
                z = q[:, 3]
                R[:, 0, 0] = 1 - 2 * (y*y + z*z)
                R[:, 0, 1] = 2 * (x*y - r*z)
                R[:, 0, 2] = 2 * (x*z + r*y)
                R[:, 1, 0] = 2 * (x*y + r*z)
                R[:, 1, 1] = 1 - 2 * (x*x + z*z)
                R[:, 1, 2] = 2 * (y*z - r*x)
                R[:, 2, 0] = 2 * (x*z - r*y)
                R[:, 2, 1] = 2 * (y*z + r*x)
                R[:, 2, 2] = 1 - 2 * (x*x + y*y)
                return R
            n_init_points = self.get_xyz.shape[0]
            padded_grad = torch.zeros((n_init_points), device=self.get_xyz.device)
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)  #extract points that satisfy the gradient condition
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means =torch.zeros((stds.size(0), 3),device=stds.device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, optimizer)
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=selected_pts_mask.device, dtype=bool)))
            prune_points(prune_filter, optimizer)

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        densify_and_clone(grads, max_grad, extent, optimizer)
        densify_and_split(grads, max_grad, extent, optimizer)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        prune_points(prune_mask, optimizer)
        torch.cuda.empty_cache()

    def reset_opacity(self, optimizer):
        def replace_tensor_to_optimizer(tensor, name, optimizer):
            optimizable_tensors = {}
            for group in optimizer.param_groups:
                if group["name"] == name:
                    stored_state = optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
            return optimizable_tensors

        def inverse_sigmoid(x): 
            return torch.log(x/(1.-x))   
     
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        self._opacity = replace_tensor_to_optimizer(opacities_new, "opacity", optimizer)["opacity"]
    
    def get_covariance(self, scaling_modifier = 1):  #render
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

class GsRender:
    def __init__(self):
        pass

    def render(self, viewpoint_camera, pc, bg_color, device, compute_cov3D_python=False, convert_SHs_python=False, override_color=None, scaling_modifier=1.0):
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device)
        screenspace_points.retain_grad()
        tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
        tanfovy = math.tan(viewpoint_camera.FovY * 0.5)
        raster_settings = GaussianRasterizationSettings(image_height=int(viewpoint_camera.image_height), 
                                                        image_width=int(viewpoint_camera.image_width), 
                                                        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color, 
                                                        scale_modifier=scaling_modifier, viewmatrix=viewpoint_camera.world_view_transform, 
                                                        projmatrix=viewpoint_camera.full_proj_transform, sh_degree=pc.now_sh_degree, 
                                                        campos=viewpoint_camera.camera_center, prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        if compute_cov3D_python:  #If precomputed 3d covariance is provided, use it. If not, then it will be computed from scaling / rotation by the rasterizer.
            scales = None
            rotations = None
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation
            cov3D_precomp = None
        if override_color is None:  #If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.now_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                shs = None
            else:
                colors_precomp = None
                shs = pc.get_features
        else:
            colors_precomp = override_color      
        # TODO 这里的means2D参数函数内好像根本没用，配起环境了之后debug看一下
        rendered_image, radii = rasterizer(means3D=means3D, means2D=means2D, shs=shs, colors_precomp=colors_precomp, 
                                           opacities=opacity, scales=scales, rotations=rotations, 
                                           cov3D_precomp=cov3D_precomp)  #rasterize visible Gaussians to image, obtain their radii (on screen). 
        return rendered_image, screenspace_points, radii, radii>0 #Those Gaussians that were frustum culled or had a radius of 0 were not visible. They will be excluded from value updates used in the splitting criteria.

def setup_wandb():
    print("Weight & Bias logging option is on.")
    username = getpass.getuser()
    # print(username)
    wandb_key_path = username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):"
        )
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system('export WANDB_API_KEY=$(cat "' + wandb_key_path + '")')

def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()

def main():
    def update_learning_rate(optimizer, iteration, position_lr_max_steps, spatial_lr_scale):
        def expon_lr(step, lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                return 0.0
            if lr_delay_steps > 0:
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        for param_group in optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = expon_lr(step=iteration, lr_init=0.00016*spatial_lr_scale, lr_final=0.0000016*spatial_lr_scale, 
                              lr_delay_steps=0, lr_delay_mult=0.01, max_steps=position_lr_max_steps)
                param_group['lr'] = lr
                return lr
            
    config_file_path = ''
    config = Gaussian_config()
    config.load(config_file_path)

    spatial_lr_scale = config.spatial_lr_scale
    position_lr_max_steps = config.position_lr_max_steps
    device = config.device

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    run_name = config.name + "_" + ts
    run_path = os.path.join(config.output_root, run_name)
    access = 0o755
    os.makedirs(run_path, access, exist_ok=True)
    assert os.access(run_path, os.W_OK)
    print(f"Start {run_path}")

    mesh_path = os.path.join(run_path, "mesh")
    image_path = os.path.join(run_path, "image")
    model_path = os.path.join(run_path, "model")
    os.makedirs(mesh_path, access, exist_ok=True)
    os.makedirs(image_path, access, exist_ok=True)
    os.makedirs(model_path, access, exist_ok=True)

    setup_wandb()
    wandb.init(project="shine_gaussian", config=vars(config), dir=run_path) # your own worksapce
    wandb.run.name = run_name
    
    shutil.copy2(config_file_path, run_path)

    print('------------init Dataset-----------------')
    gsDataset = GsDataset(device=device, pose_file_path=config.pose_file_path, image_folder_path=config.image_folder_path)
    print('------------init Network-----------------')
    gsNetwork = GsNetwork(device=device, pc_path = config.pc_path)
    gsRender = GsRender() 

    print('------------init complete-----------------')

    def ssim(img1, img2, window_size=11, size_average=True):
        def create_window(window_size, channel):
            def gaussian(window_size, sigma):
                gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
                return gauss / gauss.sum()
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
            return window
        def _ssim(img1, img2, window, window_size, channel, size_average=True):
            mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
            mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
            sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
            sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

        channel = img1.size(-3)
        window = create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        return _ssim(img1, img2, window, window_size, channel, size_average)

    xyz_lr = config.xyz_lr
    f_dc_lr = config.f_dc_lr
    f_rest_lr = config.f_rest_lr
    opacity_lr = config.opacity_lr
    scaling_lr = config.scaling_lr
    rotation_lr = config.rotation_lr
   
    optimizer = torch.optim.Adam([{'params': [gsNetwork._xyz], 'lr': xyz_lr * spatial_lr_scale, "name": "xyz"}, 
                                  {'params': [gsNetwork._features_dc], 'lr': f_dc_lr, "name": "f_dc"}, 
                                  {'params': [gsNetwork._features_rest], 'lr': f_rest_lr, "name": "f_rest"}, 
                                  {'params': [gsNetwork._opacity], 'lr': opacity_lr, "name": "opacity"}, 
                                  {'params': [gsNetwork._scaling], 'lr': scaling_lr, "name": "scaling"}, 
                                  {'params': [gsNetwork._rotation], 'lr': rotation_lr, "name": "rotation"}], lr=0.0, eps=1e-15)

    densification_interval = config.densification_interval
    opacity_reset_interval = config.opacity_reset_interval
    densify_from_iter = config.densify_from_iter
    densify_until_iter = config.densify_until_iter
    densify_grad_threshold = config.densify_grad_threshold

    out_result_iter = config.out_result_iter

    lambda_dssim = config.lambda_dssim
    white_background = config.white_background
    background = torch.tensor([[0, 0, 0],[1, 1, 1]][white_background]).float().to(device)
    viewpoint_stack = gsDataset.image_camera.copy()

    T0 = get_time()

    for iteration in range(1, position_lr_max_steps+1):
        if iteration % (position_lr_max_steps//30) == 0: gsNetwork.oneupSHdegree()

        rand_index = random.randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack[rand_index]        
        image, viewspace_point_tensor, radii,visibility_filter = gsRender.render(viewpoint_cam, gsNetwork, background, device=device)

        gt_image = viewpoint_cam.image_goal.to(device)
        Ll1 = torch.abs((image - gt_image)).mean()

        psnr = mse2psnr(img2mse(image, gt_image))

        l_ssim = ssim(image, gt_image)

        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - l_ssim)
        loss.backward() 

        with torch.no_grad():
            if iteration < densify_until_iter:                
                gsNetwork.max_radii2D[visibility_filter] = torch.max(gsNetwork.max_radii2D[visibility_filter], radii[visibility_filter])  #keep track of max radii in image-space for pruning
                gsNetwork.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opacity_reset_interval else None
                    gsNetwork.densify_and_prune(densify_grad_threshold, 0.005, gsDataset.cameras_extent, size_threshold, optimizer)                
                if iteration>0 and (iteration % opacity_reset_interval == 0 or (white_background and iteration == densify_from_iter)):
                    gsNetwork.reset_opacity(optimizer)

            optimizer.step()
            optimizer.zero_grad(set_to_none = True)

        update_learning_rate(optimizer, iteration, position_lr_max_steps, spatial_lr_scale)

        wandb_log_content = {'iteration': iteration, 'rand_index': rand_index, 'L1_loss': Ll1, 'PSNR': psnr,
                             'SSIM': l_ssim, 'total_loss': loss}
        wandb.log(wandb_log_content)

        if iteration % out_result_iter == 0: 
            T1 = get_time()
            str_tmp = 'timing(s)/every_' + str(out_result_iter) + '_iter'
            wandb_log_content = {str_tmp: T1-T0}
            wandb.log(wandb_log_content)
            T0 = T1

            print('iteration=%06d  loss=%.6f'%(iteration, loss.item()))
            # os.makedirs('./outs/', exist_ok=True)
            out_image_path = image_path + '/image_%06d_out.png'%(iteration)
            gt_image_path = image_path + '/image_%06d_gt.png'%(iteration)
            # torchvision.utils.save_image(image, './outs/image_%06d_o.png'%(iteration))
            torchvision.utils.save_image(image, out_image_path)
            # torchvision.utils.save_image(gt_image, './outs/image_%06d_t.png'%(iteration))
            torchvision.utils.save_image(gt_image, gt_image_path)

    gsNetwork.move2cpu()
    print('------------begin mesh construct-----------------')
    ms.save_mesh(gsNetwork, gsRender, mesh_shape=mesh_path + '/gs_shape.obj', mesh_texture=mesh_path + '/gs_texture.obj')

if __name__ == '__main__':  # python -Bu gs.py
    main()
