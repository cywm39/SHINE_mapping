import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import wandb
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.incre_learning import cal_feature_importance
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from model.color_SDF_decoder import color_SDF_decoder
from model.sdf_decoder import SDFDecoder
from model.color_decoder import ColorDecoder
from dataset.input_dataset import InputDataset


def run_shine_mapping_batch():

    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "Please provide the path to the config file.\nTry: python shine_batch.py xxx/xxx_config.yaml"
        )
    
    run_path = setup_experiment(config)
    shutil.copy2(sys.argv[1], run_path) # copy the config file to the result folder

    dev = config.device

    # initialize the feature octree
    sdf_octree = FeatureOctree(config, is_color=False)
    color_octree = FeatureOctree(config, is_color=True)
    # initialize the mlp decoder
    sdf_mlp = SDFDecoder(config)
    color_mlp = ColorDecoder(config)

    # load the decoder model
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        mlp.load_state_dict(loaded_model["geo_decoder"])
        print("Pretrained decoder loaded")
        freeze_model(mlp) # fixed the decoder
        if 'feature_octree' in loaded_model.keys(): # also load the feature octree  
            octree = loaded_model["feature_octree"]
            octree.print_detail()

    camera2lidar_matrix = config.camera_ext_matrix
    lidar2camera_matrix = torch.tensor(np.linalg.inv(camera2lidar_matrix), dtype=config.dtype, 
                                            device="cuda", requires_grad=config.opt_calibration)

    # dataset
    dataset = InputDataset(config, sdf_octree, color_octree, lidar2camera_matrix)

    mesher = Mesher(config, sdf_octree, color_octree, sdf_mlp, color_mlp, None)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    # Visualizer on
    if config.o3d_vis_on:
        vis = MapVisualizer()
    
    # for each frame
    print("Load, preprocess and sample data")
    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or \
            frame_id % config.every_frame != 0): 
            continue
        
        t0 = get_time()  
        # preprocess, sample data and update the octree
        dataset.process_frame(frame_id)
        t1 = get_time()
        # print("data preprocessing and sampling time (s): %.3f" %(t1 - t0))
    # print("data preprocessing and sampling time (s): %.3f" %(t1 - t0))

    # learnable parameters
    sdf_octree_feat = list(sdf_octree.parameters())
    color_octree_feat = list(color_octree.parameters())
    sdf_mlp_param = list(sdf_mlp.parameters())
    color_mlp_param = list(color_mlp.parameters())
    # learnable sigma for differentiable rendering
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0) 
    # fixed sigma for sdf prediction supervised with BCE loss
    sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale
    
    # pc_map_path = run_path + '/map/pc_map_down.ply'
    # dataset.write_merged_pc(pc_map_path)

    # initialize the optimizer
    opt = setup_optimizer(config, sdf_octree_feat, color_octree_feat, sdf_mlp_param, color_mlp_param,
                               None, sigma_size, lidar2camera_matrix)

    sdf_octree.print_detail()
    color_octree.print_detail()

    if config.normal_loss_on or config.ekional_loss_on or config.proj_correction_on or config.consistency_loss_on:
        require_gradient = True
    else:
        require_gradient = False

    # begin training
    print("Begin mapping")
    cur_base_lr = config.lr
    for iter in tqdm(range(config.iters)):
        
        T0 = get_time()
        # learning rate decay
        step_lr_decay(opt, cur_base_lr, iter, config.lr_decay_step, config.lr_iters_reduce_ratio)
        
        coord, sample_depth, ray_depth, normal_label, sem_label, weight, color_label, sdf_label = dataset.get_batch_all()

        # print(ts)

        if require_gradient:
            coord.requires_grad_(True)

        T1 = get_time()
        sdf_feature = sdf_octree.query_feature(coord)
        color_feature = color_octree.query_feature(coord) # interpolate and concat the hierachical grid features
        T2 = get_time()
        sdf_pred = sdf_mlp.predict_sdf(sdf_feature)
        color_pred = color_mlp.predict_color(color_feature) # predict the scaled sdf with the feature
        T3 = get_time()
        
        surface_mask = weight > 0

        # if config.normal_loss_on or config.ekional_loss_on:
        # use non-projective distance, gradually refined
        if require_gradient:
            g = get_gradient(coord, pred)*sigma_sigmoid
            
        if config.proj_correction_on:
            cos = torch.abs(F.cosine_similarity(g, coord - origin))
            cos[~surface_mask] = 1.0
            sdf_label = sdf_label * cos

        if config.consistency_loss_on:
            near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count,coord.shape[0]),), device=dev)
            shift_scale = config.consistency_range * config.scale # 10 cm
            random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
            coord_near = coord + random_shift 
            coord_near = coord_near[near_index, :] # only use a part of these coord to speed up
            coord_near.requires_grad_(True)
            feature_near = octree.query_feature(coord_near)
            pred_near = geo_mlp.sdf(feature_near)
            g_near = get_gradient(coord_near, pred_near)*sigma_sigmoid

        cur_loss = 0.
        # calculate the loss
        sdf_pred_copy = sdf_pred.detach()
        pred_occ = torch.sigmoid(sdf_pred_copy/sigma_size) # as occ. prob.
        # pred_ray维度: (4096, 6)
        pred_ray = pred_occ.reshape(config.bs, -1)
        # sample_depth reshape后维度: (4096, 6)
        sample_depth = sample_depth.reshape(config.bs, -1)
        color_pred = color_pred.reshape(config.bs, -1, 3)
        cdr_loss = color_depth_rendering_loss(sample_depth, pred_ray, ray_depth, color_pred, color_label, neus_on=False)
        cur_loss += cdr_loss * config.cr_loss_weight
        weight = torch.abs(weight)
        sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction) 
        cur_loss += sdf_loss
        
        # optional loss (ekional, normal, gradient consistency loss)
        eikonal_loss = 0.
        if config.ekional_loss_on:
            eikonal_loss = ((1.0 - g[surface_mask].norm(2, dim=-1)) ** 2).mean() # MSE with regards to 1  
            cur_loss += config.weight_e * eikonal_loss

        consistency_loss = 0.
        if config.consistency_loss_on:
            consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
            cur_loss += config.weight_c * consistency_loss
        
        normal_loss = 0.
        if config.normal_loss_on:
            g_direction = g / g.norm(2, dim=-1)
            normal_diff = g_direction - normal_label
            normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean() 
            cur_loss += config.weight_n * normal_loss

        T4 = get_time()

        opt.zero_grad(set_to_none=True)
        if config.opt_calibration:
            cur_loss.backward(retain_graph=True)
        else:
            cur_loss.backward()
        
        opt.step()

        T5 = get_time()

        # log to wandb
        if config.wandb_vis_on:
            if config.ray_loss:
                wandb_log_content = {'iter': iter, 'loss/total_loss': cur_loss, 
                                     'loss/color_render_loss': cdr_loss, 'loss/sdf_loss': sdf_loss, 
                                     'loss/eikonal_loss': eikonal_loss, 'loss/normal_loss': normal_loss, 
                                     'para/sigma': sigma_size.item()}
            wandb_log_content['timing(s)/load'] = T1 - T0
            wandb_log_content['timing(s)/get_indices'] = T2 - T1
            wandb_log_content['timing(s)/inference'] = T3 - T2
            wandb_log_content['timing(s)/cal_loss'] = T4 - T3
            wandb_log_content['timing(s)/back_prop'] = T5 - T4
            wandb_log_content['timing(s)/total'] = T5 - T0
            wandb.log(wandb_log_content)

        # save checkpoint model
        if (((iter+1) % config.save_freq_iters) == 0 and iter > 0):
            checkpoint_name = 'model/model_iter_' + str(iter+1)
            # octree.clear_temp()
            save_checkpoint(sdf_octree, color_octree, sdf_mlp, color_mlp, opt, run_path, checkpoint_name, iter, sigma_size)
            save_decoder(sdf_mlp, color_mlp, run_path, checkpoint_name)

        # reconstruction by marching cubes
        if (((iter+1) % config.vis_freq_iters) == 0 and iter > 0): 
            print("Begin mesh reconstruction from the implicit map")    
            mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + ".ply"
            map_path = run_path + '/map/sdf_map_iter_' + str(iter+1) + ".ply"
            if config.mc_with_octree: # default
                cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, config.semantic_on)
            else:
                cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
            if config.o3d_vis_on:
                cur_mesh.transform(dataset.begin_pose_inv)
                vis.update_mesh(cur_mesh)

    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    run_shine_mapping_batch()
