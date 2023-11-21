from typing import List
import sys
import os
import multiprocessing
import getpass
import time
from pathlib import Path
from datetime import datetime
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.autograd import grad
import torch
import torch.nn as nn
import numpy as np
import wandb
import json
import open3d as o3d

from utils.config import SHINEConfig
from model.color_decoder import ColorDecoder
from model.sdf_decoder import SDFDecoder
from model.feature_octree import FeatureOctree
from utils.mesher import Mesher
from utils.pose import read_poses_file
from natsort import natsorted 


# setup this run
def setup_experiment(config: SHINEConfig): 

    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    run_name = config.name + "_" + ts  # modified to a name that is easier to index
        
    run_path = os.path.join(config.output_root, run_name)
    access = 0o755
    os.makedirs(run_path, access, exist_ok=True)
    assert os.access(run_path, os.W_OK)
    print(f"Start {run_path}")

    mesh_path = os.path.join(run_path, "mesh")
    map_path = os.path.join(run_path, "map")
    model_path = os.path.join(run_path, "model")
    os.makedirs(mesh_path, access, exist_ok=True)
    os.makedirs(map_path, access, exist_ok=True)
    os.makedirs(model_path, access, exist_ok=True)
    
    if config.wandb_vis_on:
        # set up wandb
        setup_wandb()
        wandb.init(project="SHINEMapping", config=vars(config), dir=run_path) # your own worksapce
        wandb.run.name = run_name         
    
    # set the random seed (for deterministic experiment results)
    o3d.utility.random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed) 

    return run_path


# 在默认的配置下，只有octree和geo_mlp被加入到optimizer中
def setup_optimizer(config: SHINEConfig, sdf_octree_feat, color_octree_feat, sdf_mlp_param, color_mlp_param, 
                    mlp_sem_param, sigma_size, lidar2camera_matrix) -> Optimizer:
    lr_cur = config.lr
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    if sdf_mlp_param is not None:
        sdf_mlp_param_opt_dict = {'params': sdf_mlp_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(sdf_mlp_param_opt_dict)
    if color_mlp_param is not None:
        color_mlp_param_opt_dict = {'params': color_mlp_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(color_mlp_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {'params': mlp_sem_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_sem_param_opt_dict)
    if lidar2camera_matrix is not None:
        lidar2camera_matrix_param_opt_dict = {'params': [lidar2camera_matrix], 'lr': config.calibration_lr, 'weight_decay': config.weight_decay}
        opt_setting.append(lidar2camera_matrix_param_opt_dict)
    # feature octree
    for i in range(config.tree_level_feat):
        # try to also add L2 regularization on the feature octree (results not quite good)
        # from bottom to top遍历octree中带有feature的层，最底层lr是config中设置的lr，往上每层多乘以一次lr_level_reduce_ratio
        feat_opt_dict = {'params': sdf_octree_feat[config.tree_level_feat-i-1], 'lr': lr_cur} 
        lr_cur *= config.lr_level_reduce_ratio
        opt_setting.append(feat_opt_dict)
    lr_cur = config.lr
    for i in range(config.color_tree_level_feat):
        # try to also add L2 regularization on the feature octree (results not quite good)
        # from bottom to top遍历octree中带有feature的层，最底层lr是config中设置的lr，往上每层多乘以一次lr_level_reduce_ratio
        feat_opt_dict = {'params': color_octree_feat[config.color_tree_level_feat-i-1], 'lr': lr_cur} 
        lr_cur *= config.lr_level_reduce_ratio
        opt_setting.append(feat_opt_dict)
    # make sigma also learnable for differentiable rendering (but not for our method)
    if config.ray_loss:
        sigma_opt_dict = {'params': sigma_size, 'lr': config.lr}
        opt_setting.append(sigma_opt_dict)
    
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt


def setup_optimizer_traj(config: SHINEConfig, octree_feat, mlp_geo_param, mlp_sem_param, mlp_traj_param, sigma_size) -> Optimizer:
    lr_cur = config.lr
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    if mlp_geo_param is not None:
        mlp_geo_param_opt_dict = {'params': mlp_geo_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_geo_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {'params': mlp_sem_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_sem_param_opt_dict)
    if mlp_traj_param is not None:
        mlp_traj_param_opt_dict = {'params': mlp_traj_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_traj_param_opt_dict)

    # feature octree
    for i in range(config.tree_level_feat):
        # try to also add L2 regularization on the feature octree (results not quite good)
        feat_opt_dict = {'params': octree_feat[config.tree_level_feat-i-1], 'lr': lr_cur} 
        lr_cur *= config.lr_level_reduce_ratio
        opt_setting.append(feat_opt_dict)
    # make sigma also learnable for differentiable rendering (but not for our method)
    if config.ray_loss:
        sigma_opt_dict = {'params': sigma_size, 'lr': config.lr}
        opt_setting.append(sigma_opt_dict)
    
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt
    

# set up weight and bias
def setup_wandb():
    print("Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file.")
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

def step_lr_decay(
    optimizer: Optimizer,
    learning_rate: float,
    iteration_number: int,
    steps: List,
    reduce: float = 1.0):

    if reduce > 1.0 or reduce <= 0.0:
        sys.exit(
            "The decay reta should be between 0 and 1."
        )

    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate


def num_model_weights(model: nn.Module) -> int:
    num_weights = int(
        sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        )
    )
    return num_weights


def print_model_summary(model: nn.Module):
    for child in model.children():
        print(child)


def get_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad


def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True


def save_checkpoint(
    sdf_octree, color_octree, sdf_decoder, color_decoder, optimizer, run_path, checkpoint_name, frame
):
    torch.save(
        {
            "frame": frame,
            "sdf_octree": sdf_octree, # save the whole NN module (the hierachical features and the indexing structure)
            "color_octree": color_octree,
            "sdf_decoder": sdf_decoder.state_dict(),
            "color_decoder": color_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(run_path, f"{checkpoint_name}.pth"),
    )
    print(f"save the model to {run_path}/{checkpoint_name}.pth")


def save_decoder(sdf_decoder, color_decoder, run_path, checkpoint_name):
    torch.save({"sdf_decoder": sdf_decoder.state_dict(), 
                "color_decoder": color_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_decoders.pth"),
    )

def save_geo_decoder(geo_decoder, run_path, checkpoint_name):
    torch.save({"geo_decoder": geo_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_geo_decoder.pth"),
    )

def save_sem_decoder(sem_decoder, run_path, checkpoint_name):
    torch.save({"sem_decoder": sem_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_sem_decoder.pth"),
    )

def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.
    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.
    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)

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


def load_model_and_recon_octree_mesh(config_file_path: str, load_model_path: str,
                                     mesh_path: str):
    config = SHINEConfig()
    config.load(config_file_path)
    sdf_octree = FeatureOctree(config, is_color=False)
    color_octree = FeatureOctree(config, is_color=True)
    sdf_mlp = SDFDecoder(config)
    color_mlp = ColorDecoder(config)

    loaded_model = torch.load(load_model_path)
    sdf_mlp.load_state_dict(loaded_model["sdf_decoder"])
    color_mlp.load_state_dict(loaded_model["color_decoder"])
    if 'sdf_octree' in loaded_model.keys(): # also load the feature octree  
        sdf_octree = loaded_model["sdf_octree"]
        sdf_octree.print_detail()

    if 'color_octree' in loaded_model.keys(): # also load the feature octree  
        color_octree = loaded_model["color_octree"]
        color_octree.print_detail()

    mesher = Mesher(config, sdf_octree, color_octree, sdf_mlp, color_mlp, None)
    begin_pose_inv = np.eye(4)
    mesher.global_transform = np.linalg.inv(begin_pose_inv)

    #  # visualize the octree (it is a bit slow and memory intensive for the visualization)
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
    cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, config.semantic_on, False)

def load_model_and_recon_bbx_mesh(config_file_path: str, load_model_path: str,
                                     mesh_path: str, map_path: str, lidar2camera_matrix):
    config = SHINEConfig()
    config.load(config_file_path)
    sdf_octree = FeatureOctree(config, is_color=False)
    color_octree = FeatureOctree(config, is_color=True)
    sdf_mlp = SDFDecoder(config)
    color_mlp = ColorDecoder(config)

    loaded_model = torch.load(load_model_path)
    sdf_mlp.load_state_dict(loaded_model["sdf_decoder"])
    color_mlp.load_state_dict(loaded_model["color_decoder"])
    if 'sdf_octree' in loaded_model.keys(): # also load the feature octree  
        sdf_octree = loaded_model["sdf_octree"]
        sdf_octree.print_detail()

    if 'color_octree' in loaded_model.keys(): # also load the feature octree  
        color_octree = loaded_model["color_octree"]
        color_octree.print_detail()

    mesher = Mesher(config, sdf_octree, color_octree, sdf_mlp, color_mlp, None)
    begin_pose_inv = np.eye(4)
    mesher.global_transform = np.linalg.inv(begin_pose_inv)

    map_down_pc = o3d.geometry.PointCloud()
    map_bbx = o3d.geometry.AxisAlignedBoundingBox()

    calib = {}
    calib['Tr'] = np.eye(4)
    poses = read_poses_file(config.pose_path, calib)

    pc_filenames = natsorted(os.listdir(config.pc_path))
    pose_index = 0
    pc_radius = config.pc_radius
    min_z = config.min_z
    max_z = config.max_z
    vox_down_m = config.vox_down_m

    for filename in pc_filenames:
        frame_path = os.path.join(config.pc_path, filename)
        frame_pc = read_point_cloud(frame_path)


        bbx_min = np.array([-pc_radius, -pc_radius, min_z])
        bbx_max = np.array([pc_radius, pc_radius, max_z])
        bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
        frame_pc = frame_pc.crop(bbx)

        frame_pc = frame_pc.voxel_down_sample(voxel_size=vox_down_m)
        
        lidar2camera_matrix_tmp = lidar2camera_matrix
        # lidar2camera_matrix_tmp.requires_grad = False
        # 去掉不能映射到相机图片中的点
        # 将点从雷达坐标系转到相机坐标系
        frame_pc_points = np.asarray(frame_pc.points, dtype=np.float64)
        points3d_lidar = np.asarray(frame_pc.points, dtype=np.float64)
        # points3d_lidar = frame_pc.clone()
        points3d_lidar = np.insert(points3d_lidar, 3, 1, axis=1)
        points3d_camera = lidar2camera_matrix_tmp @ points3d_lidar.T
        H, W, fx, fy, cx, cy, = config.H, config.W, config.fx, config.fy, config.cx, config.cy
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
        frame_pc.points = o3d.utility.Vector3dVector(frame_pc_points)

        frame_pc = frame_pc.transform(poses[pose_index])
        frame_pc = frame_pc.voxel_down_sample(voxel_size=config.map_vox_down_m)
        map_down_pc += frame_pc

        pose_index += 1

    map_bbx = map_down_pc.get_axis_aligned_bounding_box()

    cur_mesh = mesher.recon_bbx_mesh(map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

