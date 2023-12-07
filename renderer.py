import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from utils.config import SHINEConfig
from utils.config import SHINEConfig
from model.feature_octree import FeatureOctree
from model.sdf_decoder import SDFDecoder
from model.color_decoder import ColorDecoder
from utils.pose import read_poses_file

def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    tran = np.array([[0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0,0,0,1]])
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device).float()
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    # dirs = torch.stack(
    #     [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = torch.stack(
        [-(i-cx)/fx, -(j-cy)/fy, torch.ones_like(i)], -1).to(device)

    # dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    
    dirs = dirs.reshape(-1, 3)
    rays_d = c2w[:3, :3] @ dirs.T
    rays_d = rays_d.T

    rays_d = rays_d.cpu().numpy()
    rays_d = np.insert(rays_d, 3, 1, axis=1)
    rays_d = rays_d @ tran
    rays_d = rays_d[:, :-1]
    rays_d = torch.from_numpy(rays_d).to(device)

    rays_d = rays_d.reshape(H, W ,3)

    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_bp(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    tran = np.array([[0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0,0,0,1]])
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    # dirs = torch.stack(
    #     [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = torch.stack(
        [(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1).to(device)

    # dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    
    dirs = dirs.reshape(-1, 3)
    dirs = dirs.cpu().numpy()
    dirs = np.insert(dirs, 3, 1, axis=1)
    dirs = tran @ dirs.T
    dirs = dirs.T
    dirs = dirs[:, :-1]

    rays_d = c2w[:3, :3] @ dirs.T
    rays_d = rays_d.T

    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device).float()

    # rays_d = rays_d.cpu().numpy()
    # rays_d = np.insert(rays_d, 3, 1, axis=1)
    # rays_d = rays_d @ tran
    # rays_d = rays_d[:, :-1]
    rays_d = torch.from_numpy(rays_d).to(device)

    rays_d = rays_d.reshape(H, W ,3)

    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples

def render_batch_ray(config: SHINEConfig, sdf_octree, color_octree, sdf_mlp, color_mlp,
                     rays_d, rays_o, device, sigma_size, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """

        N_samples = config.N_samples
        N_surface = config.N_surface
        N_importance = config.N_importance

        N_rays = rays_o.shape[0]

        if gt_depth is None:
            N_surface = 0
            near = 0.01
        else:
            gt_depth = gt_depth.reshape(-1, 1)
            gt_depth_samples = gt_depth.repeat(1, N_samples)
            near = gt_depth_samples*0.01

        nodes_coord_scaled = sdf_octree.get_octree_nodes(config.tree_level_world) # query level top-down
        min_nodes = np.min(nodes_coord_scaled, 0) # 最小和最大坐标点
        max_nodes = np.max(nodes_coord_scaled, 0)
        print("min_nodes: ")
        print(min_nodes)
        print("max_nodes: ")
        print(max_nodes)
        # bound = max_nodes - min_nodes
        bound = np.zeros((3,2))
        bound[:, 0] = min_nodes
        bound[:, 1] = max_nodes
        bound = bound / config.scale
        bound = torch.tensor(bound, device=device)
        print("bound: ")
        print(bound)

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (bound.unsqueeze(0).to(device) -
                 det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
        else:
            far = far_bb
        if N_surface > 0:
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
            else:
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.
                gt_none_zero_mask = gt_depth > 0
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).double().to(device)
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(
                    gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask,
                               :] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = near_surface * \
                    (1.-t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(
                    0).repeat((~gt_none_zero_mask).sum(), 1)
                z_vals_surface[~gt_none_zero_mask,
                               :] = z_vals_surface_depth_zero

        t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

        if True:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        if False:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        pointsf = pts.reshape(-1, 3)
        pointsf = pointsf * config.scale

        sdf_feature = sdf_octree.query_feature(pointsf)
        color_feature = color_octree.query_feature(pointsf)
        sdf_pred = sdf_mlp.predict_sdf(sdf_feature)
        color_pred = color_mlp.predict_color(color_feature)
        
        color_norm = np.clip(color_pred.detach().cpu().numpy(), 0, 255)
        color_norm /= 255.0
        color_pred = torch.tensor(color_norm, device=device)

        pred_occ = torch.sigmoid(sdf_pred/sigma_size)
        pred_ray = pred_occ.reshape(N_rays, -1)
        color_pred = color_pred.reshape(N_rays, -1, 3)

        alpha = pred_ray
        one_minus_alpha = torch.ones_like(alpha) - alpha + 1e-10
        cum_mat = torch.cumprod(one_minus_alpha, 1)
        weights = cum_mat / one_minus_alpha * alpha

        weights_tmp = weights.unsqueeze(2).expand(-1, -1, 3)
        weights_color = weights_tmp * color_pred[:, 0 : alpha.shape[1], :]
        color_render = torch.sum(weights_color, 1).squeeze(1)

        depth = torch.sum(weights * z_vals, -1)

        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=True, device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + \
                rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            pts = pts * config.scale

            sdf_feature = sdf_octree.query_feature(pts)
            color_feature = color_octree.query_feature(pts)
            sdf_pred = sdf_mlp.predict_sdf(sdf_feature)
            color_pred = color_mlp.predict_color(color_feature)

            color_norm = np.clip(color_pred.detach().cpu().numpy(), 0, 255)
            color_norm /= 255.0
            color_pred = torch.tensor(color_norm, device=device)

            pred_occ = torch.sigmoid(sdf_pred/sigma_size)
            pred_ray = pred_occ.reshape(N_rays, -1)
            color_pred = color_pred.reshape(N_rays, -1, 3)

            alpha = pred_ray
            one_minus_alpha = torch.ones_like(alpha) - alpha + 1e-10
            cum_mat = torch.cumprod(one_minus_alpha, 1)
            weights = cum_mat / one_minus_alpha * alpha

            weights_tmp = weights.unsqueeze(2).expand(-1, -1, 3)
            weights_color = weights_tmp * color_pred[:, 0 : alpha.shape[1], :]
            color_render = torch.sum(weights_color, 1).squeeze(1)

            depth = torch.sum(weights * z_vals, -1)
    
            return depth, color_render, pts

        return depth, color_render, pointsf


        
def render_img(config: SHINEConfig, sdf_octree, color_octree, sdf_mlp, color_mlp, L2w, sigma_size, device):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            camera2lidar_matrix = config.camera_ext_matrix
            lidar2camera_matrix = np.linalg.inv(camera2lidar_matrix)
            c2w = L2w @ lidar2camera_matrix

            H = config.H
            W = config.W
            rays_o, rays_d = get_rays(
                H, W, config.fx, config.fy, config.cx, config.cy, c2w, device)
            # rays_o, rays_d = calculate_camera_ray_params(
            #     H, W, config.fx, config.fy, config.cx, config.cy, c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # get depth map from point cloud map
            pc_map = o3d.io.read_point_cloud(config.pc_map_path)
            pc_map = pc_map.transform(np.linalg.inv(L2w))
            points3d_lidar = np.asarray(pc_map.points, dtype=np.float64)
            points3d_lidar = np.insert(points3d_lidar, 3, 1, axis=1)
            points3d_lidar = points3d_lidar @ np.array([[0, 0, 1, 0],
                                                [-1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0,0,0,1]]).reshape(4,4)
            points3d_camera = lidar2camera_matrix @ points3d_lidar.T
            H, W, fx, fy, cx, cy, = config.H, config.W, config.fx, config.fy, config.cx, config.cy
            K = np.array([[fx, .0, cx, .0], [.0, fy, cy, .0], [.0, .0, 1.0, .0]]).reshape(3, 4)
            # 过滤掉相机坐标系内位于相机之后的点
            tmp_mask = points3d_camera[2, :] > 0.0
            points3d_camera = points3d_camera[:, tmp_mask]
            points3d_camera_tmp = np.copy(points3d_camera)
            points2d_camera = K @ points3d_camera
            points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 操作之后points2d_camera维度:[n, 2]
            tmp_mask = np.logical_and(
                (points2d_camera[:, 1] < H) & (points2d_camera[:, 1] > 0),
                (points2d_camera[:, 0] < W) & (points2d_camera[:, 0] > 0)
            )
            points2d_camera = points2d_camera[tmp_mask]
            points3d_camera_tmp = points3d_camera_tmp[:, tmp_mask]
            depth_image = np.zeros((H, W))
            test_image = np.zeros((H, W, 3))
            depth_image[points2d_camera[:,1].astype(int), points2d_camera[:,0].astype(int)] = points3d_camera_tmp[2, :]
            test_image[points2d_camera[:,1].astype(int), points2d_camera[:,0].astype(int)] = np.array([255,255,255])

            depth_test = Image.fromarray((depth_image).astype(np.uint8))  # 假设深度值范围为 [0, 1]
            color_test = Image.fromarray((test_image).astype(np.uint8))  # 假设颜色值范围为 [0, 1]
            depth_test.save('./test_result/test_depth.png')
            color_test.save('./test_result/test_color.png')
            
            gt_depth = depth_image

            gt_depth = gt_depth.reshape(-1)
            gt_depth = torch.tensor(gt_depth, device=device)

            depth_list = []
            uncertainty_list = []
            color_list = []
            points_list = []

            ray_batch_size = config.ray_batch_size

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                # if True:
                    ret = render_batch_ray(
                        config, sdf_octree, color_octree, sdf_mlp, color_mlp, 
                        rays_d_batch, rays_o_batch, device, sigma_size, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = render_batch_ray(
                        config, sdf_octree, color_octree, sdf_mlp, color_mlp, 
                        rays_d_batch, rays_o_batch, device, sigma_size, gt_depth=gt_depth_batch)

                depth, color, points_batch = ret
                depth_list.append(depth.double())
                color_list.append(color)
                points_list.append(points_batch)

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)
            points = torch.cat(points_list, dim=0)
            points = points.cpu().numpy()

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = pcd.scale(1.0 / config.scale, center=(0,0,0))

            return depth, color, pcd


def render_from_pc(config: SHINEConfig, sdf_octree, color_octree, sdf_mlp, color_mlp, L2w, sigma_size, device):
    pc_map = o3d.io.read_point_cloud(config.pc_map_path)
    pc_map = pc_map.transform(np.linalg.inv(L2w))
    points3d_lidar = np.asarray(pc_map.points, dtype=np.float64)
    points3d_lidar = np.insert(points3d_lidar, 3, 1, axis=1)
    points3d_camera = lidar2camera_matrix @ points3d_lidar.T
    H, W, fx, fy, cx, cy, = config.H, config.W, config.fx, config.fy, config.cx, config.cy
    K = np.array([[fx, .0, cx, .0], [.0, fy, cy, .0], [.0, .0, 1.0, .0]]).reshape(3, 4)
    # 过滤掉相机坐标系内位于相机之后的点
    tmp_mask = points3d_camera[2, :] > 0.0
    points3d_camera = points3d_camera[:, tmp_mask]
    points3d_camera_tmp = np.copy(points3d_camera)
    points2d_camera = K @ points3d_camera
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 操作之后points2d_camera维度:[n, 2]
    tmp_mask = np.logical_and(
        (points2d_camera[:, 1] < H) & (points2d_camera[:, 1] > 0),
        (points2d_camera[:, 0] < W) & (points2d_camera[:, 0] > 0)
    )
    points2d_camera = points2d_camera[tmp_mask]
    points3d_camera_tmp = points3d_camera_tmp[:, tmp_mask]
    depth_image = np.zeros((H, W))
    test_image = np.zeros((H, W, 3))
    depth_image[points2d_camera[:,1].astype(int), points2d_camera[:,0].astype(int)] = points3d_camera_tmp[2, :]
    test_image[points2d_camera[:,1].astype(int), points2d_camera[:,0].astype(int)] = np.array([255,255,255])

    depth_test = Image.fromarray((depth_image).astype(np.uint8))  # 假设深度值范围为 [0, 1]
    color_test = Image.fromarray((test_image).astype(np.uint8))  # 假设颜色值范围为 [0, 1]
    depth_test.save('./test_result/test_depth.png')
    color_test.save('./test_result/test_color.png')


def render_color_pc(config: SHINEConfig, sdf_octree, color_octree, sdf_mlp, color_mlp, device):
    pc_map = o3d.io.read_point_cloud(config.pc_map_path)
    pc_map = pc_map.scale(config.scale, center=(0,0,0))
    points3d = np.asarray(pc_map.points, dtype=np.float64)
    color = np.ones_like(points3d)
    # color_points[:, 3:] = np.array([57,197,187])

    for i in range(0, points3d.shape[0], config.point_batch_size):
        points_batch = points3d[i:i+config.point_batch_size]
        points_batch = torch.tensor(points_batch, device=device)

        sdf_feature = sdf_octree.query_feature(points_batch)
        color_feature = color_octree.query_feature(points_batch)
        sdf_pred = sdf_mlp.predict_sdf(sdf_feature)
        color_pred = color_mlp.predict_color(color_feature)

        color[i:i+config.point_batch_size] = color_pred.detach().cpu().numpy()

    color = np.clip(color, 0, 255)
    color /= 255.0
    # 将numpy数组转换为Open3D的点云数据结构
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)  # 提取点的坐标
    pcd.colors = o3d.utility.Vector3dVector(color)  # 提取点的颜色
    pcd = pcd.scale(1.0 / config.scale, center=(0,0,0))

    return pcd
    

if __name__ == "__main__":
    config = SHINEConfig()
    config.load('/home/wuchenyang/NeRF/SHINE_mapping/config/carla/rgb_carla_batch.yaml')

    sdf_octree = FeatureOctree(config, is_color=False)
    color_octree = FeatureOctree(config, is_color=True)

    sdf_mlp = SDFDecoder(config)
    color_mlp = ColorDecoder(config)

    loaded_model = torch.load('/home/wuchenyang/NeRF/SHINE_mapping/experiments/rgb_carla_batch_2023-11-24_00-49-02/model/model_iter_100000.pth')
    sdf_mlp.load_state_dict(loaded_model["sdf_decoder"])
    color_mlp.load_state_dict(loaded_model["color_decoder"])

    if 'sdf_octree' in loaded_model.keys(): # also load the feature octree  
        sdf_octree = loaded_model["sdf_octree"]
        sdf_octree.print_detail()

    if 'color_octree' in loaded_model.keys(): # also load the feature octree  
        color_octree = loaded_model["color_octree"]
        color_octree.print_detail()

    if 'sigma_size' in loaded_model.keys():
        sigma_size = loaded_model['sigma_size']
        # sigma_size = torch.tensor(sigma_size, device="cuda")
        sigma_size = sigma_size.to(device='cuda')
        print('loaded sigma_size: ')
        print(sigma_size)
    else:
        sigma_size = torch.ones(1, device="cuda")*3.8445
        print('setting sigma_size: ')
        print(sigma_size)

    # #-------------------------------render a pc with color---------------------------------
    # color_pc = render_color_pc(config, sdf_octree, color_octree, sdf_mlp, color_mlp, 'cuda')
    # o3d.io.write_point_cloud("./test_result/colored_point_cloud.pcd", color_pc)
    # #-------------------------------render a pc with color---------------------------------

    calib = {}
    calib['Tr'] = np.eye(4)
    poses = read_poses_file(config.pose_path, calib)    

    tran = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0,0,0,1]])

    pose =  poses[400] 

    # #--------------------test--------------------------
    # print(pose)
    # camera2lidar_matrix = config.camera_ext_matrix
    # lidar2camera_matrix = np.linalg.inv(camera2lidar_matrix)
    # c2w = pose @ lidar2camera_matrix

    # H = config.H
    # W = config.W
    # rays_o, rays_d = get_rays(
    #     H, W, config.fx, config.fy, config.cx, config.cy, c2w, 'cuda')
    # rays_o = rays_o.reshape(-1, 3)
    # rays_d = rays_d.reshape(-1, 3)
    # t_vals = torch.linspace(0., 1., steps=32, device='cuda')
    # z_vals = 0.1 * (1.-t_vals) + 20 * (t_vals)
    # pts = rays_o[..., None, :] + rays_d[..., None, :] * \
    #     z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
    # pointsf = pts.reshape(-1, 3)
    # pointsf = pointsf.cpu().numpy()
    # # pointsf = np.insert(pointsf, 3, 1, axis=1)
    # # pointsf = pointsf @ tran
    # # pointsf = pointsf[:, :-1]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointsf)
    # o3d.io.write_point_cloud("./test_result/ray_test_point_cloud.pcd", pcd)
    # #--------------------test--------------------------

    depth_img, color_img, sample_pc = render_img(config, sdf_octree, color_octree, sdf_mlp, color_mlp,
                                      pose, sigma_size, "cuda")
    
    depth_array = depth_img.cpu().numpy()
    color_array = color_img.cpu().numpy()

    # 创建 PIL Image 对象
    depth_image = Image.fromarray((depth_array * 255).astype(np.uint8))  # 假设深度值范围为 [0, 1]
    color_image = Image.fromarray((color_array * 255).astype(np.uint8))  # 假设颜色值范围为 [0, 1]

    # 保存图像
    depth_image.save('./test_result/depth_image.png')
    color_image.save('./test_result/color_image.png')
    o3d.io.write_point_cloud("./test_result/sample_point_cloud.pcd", sample_pc)
