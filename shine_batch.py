import sys
import wandb
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import open3d as o3d
import wandb
import torch
import torch.nn as nn
import torch.optim as optim


from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.mesher import Mesher
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset


def main():

    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "Please provide the path to the config file.\nTry: python shine_batch.py xxx/xxx_config.yaml"
        )
    
    run_path = setup_experiment(config)
    dev = config.device

    # initialize the feature octree
    octree = FeatureOctree(config)
    # initialize the mlp decoder
    mlp = Decoder(config)

    # dataset
    dataset = LiDARDataset(config, octree)

    mesher = Mesher(config, octree, mlp)
    
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

    # learnable parameters
    octree_feat = list(octree.parameters())
    mlp_param = list(mlp.parameters())
    # learnable sigma for differentiable rendering
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0) 
    # fixed sigma for sdf prediction supervised with BCE loss
    sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale
    
    # initialize the optimizer
    opt = setup_optimizer(config, octree_feat, mlp_param, sigma_size)

    octree.print_detail()

    # begin training
    print("Begin mapping")
    cur_base_lr = config.lr
    for iter in tqdm(range(config.iters)):
        
        T0 = get_time()
        # learning rate decay
        step_lr_decay(opt, cur_base_lr, iter, config.lr_decay_step, config.lr_iters_reduce_ratio)
        
        # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)
        if config.ray_loss:       
            coord, sample_depth, ray_depth, normal_label, weight = dataset.get_batch()
        else:
            coord, sdf_label, normal_label, weight = dataset.get_batch()

        T1 = get_time()
                
        octree.get_indices(coord)
        if config.normal_loss_on or config.ekional_loss_on:
            coord.requires_grad_()
            
        T2 = get_time()

        feature = octree.query_feature(coord) # interpolate and concat the hierachical grid features
        pred = mlp(feature) # predict the scaled sdf with the feature
                
        T3 = get_time()
        
        surface_mask = weight > 0
        cur_loss = 0.
        # calculate the loss
        if config.ray_loss: # neural rendering loss       
            pred = torch.sigmoid(pred/sigma_size) 
            pred_ray = pred.reshape(config.bs, -1)
            sample_depth = sample_depth.reshape(config.bs, -1)
            if config.main_loss_type == "dr":
                dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=False)
            elif config.main_loss_type == "dr_neus":
                dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=True)
            cur_loss += dr_loss
        else: # sdf regression loss
            weight = torch.abs(weight) # weight's sign indicate the sample is around the surface or in the free space
            if config.main_loss_type == "sdf_bce": # our proposed method
                sdf_loss = sdf_bce_loss(pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on) 
            elif config.main_loss_type == "sdf_l1": 
                sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=False)
            elif config.main_loss_type == "sdf_l2":
                sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=True)
            cur_loss += sdf_loss
        
        # optional loss (ekional, normal loss)
        if config.normal_loss_on or config.ekional_loss_on:
            g = gradient(coord, pred)*sigma_sigmoid
        eikonal_loss = 0.
        if config.ekional_loss_on:
            eikonal_loss = ((g[surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # MSE with regards to 1  
            cur_loss += config.weight_e * eikonal_loss
        normal_loss = 0.
        if config.normal_loss_on:
            normal_diff = g - normal_label
            normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean() 
            cur_loss += config.weight_n * normal_loss

        T4 = get_time()

        opt.zero_grad(set_to_none=True)
        cur_loss.backward()
        opt.step()
        octree.set_zero() # set the trashbin feature vector back to 0 after the feature update

        T5 = get_time()

        # log to wandb
        if config.wandb_vis_on:
            if config.ray_loss:
                wandb_log_content = {'iter': iter, 'loss/total_loss': cur_loss, 'loss/render_loss': dr_loss, 'loss/eikonal_loss': eikonal_loss, 'loss/normal_loss': normal_loss, 'para/sigma': sigma_size} 
            else:
                wandb_log_content = {'iter': iter, 'loss/total_loss': cur_loss, 'loss/sdf_loss': sdf_loss, 'loss/eikonal_loss': eikonal_loss, 'loss/normal_loss': normal_loss} 
            wandb_log_content['timing(s)/load'] = T1 - T0
            wandb_log_content['timing(s)/get_indices'] = T2 - T1
            wandb_log_content['timing(s)/inference'] = T3 - T2
            wandb_log_content['timing(s)/cal_loss'] = T4 - T3
            wandb_log_content['timing(s)/back_prop'] = T5 - T4
            wandb_log_content['timing(s)/total'] = T5 - T0
            wandb.log(wandb_log_content)

        # save checkpoint model
        if ((iter % config.save_freq_iters) == 0 and iter > 0):
            checkpoint_name = 'model/model_iter_' + str(iter)
            save_checkpoint(octree, mlp, opt, run_path, checkpoint_name, iter)

        # reconstruction by marching cubes
        if ((iter % config.vis_freq_iters) == 0 and iter > 0): 
            # print("Begin reconstruction from implicit map")               
            mesh_path = run_path + '/mesh/mesh_iter_' + str(iter) + ".ply"
            mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path)

        # evaluation

    


if __name__ == "__main__":
    main()