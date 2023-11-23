"""
author: cy
"""
import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import open3d as o3d
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import shutil

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

def run_shine_mapping_incremental():
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(sci_mode=False, precision = 10)


    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "Please provide the path to the config file.\nTry: python shine_incre.py xxx/xxx_config.yaml"
        )

    run_path = setup_experiment(config)
    shutil.copy2(sys.argv[1], run_path) # copy the config file to the result folder
    
    dev = config.device

    # initialize the feature octree
    sdf_octree = FeatureOctree(config, is_color=False)
    color_octree = FeatureOctree(config, is_color=True)
    # initialize the mlp decoder

    # mlp = color_SDF_decoder(config)
    sdf_mlp = SDFDecoder(config)
    color_mlp = ColorDecoder(config)

    # Load the decoder model
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        mlp.load_state_dict(loaded_model["geo_decoder"])
        print("Pretrained decoder loaded")
        freeze_model(mlp) # fixed the decoder
        if 'feature_octree' in loaded_model.keys(): # also load the feature octree  
            octree = loaded_model["feature_octree"]
            octree.print_detail()

    camera2lidar_matrix = config.camera_ext_matrix
    lidar2camera_matrix = nn.Parameter(torch.tensor(np.linalg.inv(camera2lidar_matrix), dtype=config.dtype, device="cuda", requires_grad=True))
    # lidar2camera_matrix.retain_grad()
    # dataset
    dataset = InputDataset(config, sdf_octree, color_octree, lidar2camera_matrix)

    # mesh reconstructor
    mesher = Mesher(config, sdf_octree, color_octree, sdf_mlp, color_mlp, None)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    # Non-blocking visualizer
    if config.o3d_vis_on:
        vis = MapVisualizer()

    # learnable parameters
    # mlp_param = list(mlp.parameters())
    sdf_mlp_param = list(sdf_mlp.parameters())
    color_mlp_param = list(color_mlp.parameters())
    # learnable sigma for differentiable rendering
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0) 
    # fixed sigma for sdf prediction supervised with BCE loss
    sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale

    processed_frame = 0
    total_iter = 0
    if config.continual_learning_reg:
        config.loss_reduction = "sum" # other-wise "mean"

    if config.normal_loss_on or config.ekional_loss_on or config.proj_correction_on or config.consistency_loss_on:
        require_gradient = True
    else:
        require_gradient = False

    # for each frame
    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or \
            frame_id % config.every_frame != 0): 
            continue
        
        vis_mesh = False 

        if processed_frame == config.freeze_after_frame: # freeze the decoder after certain frame
            print("Freeze the decoder")
            freeze_model(sdf_mlp) # fixed the decoder
            freeze_model(color_mlp)

        T0 = get_time()
        # preprocess, sample data and update the octree
        # if continual_learning_reg is on, we only keep the current frame's sample in the data pool,
        # otherwise we accumulate the data pool with the current frame's sample

        local_data_only = True # this one would lead to the forgetting issue. default: False

        # 读取新的一帧点云并预处理，对这一帧点云进行采样得到采样点，更新octree，更新data pool
        dataset.process_frame(frame_id, incremental_on=config.continual_learning_reg or local_data_only)
        
        sdf_octree_feat = list(sdf_octree.parameters())
        color_octree_feat = list(color_octree.parameters())
        # 每帧都会设置一次optimizer，原因是每帧都会更新octree
        opt = setup_optimizer(config, sdf_octree_feat, color_octree_feat, sdf_mlp_param, color_mlp_param,
                               None, sigma_size, lidar2camera_matrix)
        sdf_octree.print_detail()
        color_octree.print_detail()
        lidar2camera_matrix_backup = lidar2camera_matrix.detach().clone()

        T1 = get_time()

        # get_batch随机选择batch，输入octree得到feature，输入mlp得到sdf，和通过了sigmoid的label计算loss，再加点其他loss，backward和step
        for iter in tqdm(range(config.iters)):
            # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)

            # 必须用ray_loss的get_batch
            coord, sample_depth, ray_depth, normal_label, sem_label, weight, color_label, sdf_label = dataset.get_batch_all()
            
            if require_gradient:
                coord.requires_grad_(True)

            # interpolate and concat the hierachical grid features
            # 这里已经concat了各个level上的feature
            sdf_feature = sdf_octree.query_feature(coord)
            color_feature = color_octree.query_feature(coord)
            
            # predict the scaled sdf with the feature
            # 输入的feature维度是(n, 8)，返回的sdf_pred维度是(n, 1)
            # sdf_pred, color_pred = mlp(sdf_feature, color_feature)
            sdf_pred = sdf_mlp.predict_sdf(sdf_feature)
            color_pred = color_mlp.predict_color(color_feature)

            # calculate the loss
            surface_mask = weight > 0

            if require_gradient:
                # g在下面求eikonal_loss和consistency_loss的时候用到了，前者的公式里需要的就是网络输出对输入点坐标的偏导，所以这里的inputs才设置成coord
                # 一般用到torch.autograd.grad的时候inputs都指定成模型参数的
                g = get_gradient(coord, sdf_pred)*sigma_sigmoid

            # if config.consistency_loss_on:
            #     near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count,coord.shape[0]),), device=dev)
            #     shift_scale = config.consistency_range * config.scale # 10 cm
            #     random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
            #     coord_near = coord + random_shift 
            #     coord_near = coord_near[near_index, :] # only use a part of these coord to speed up
            #     coord_near.requires_grad_(True)
            #     feature_near = octree.query_feature(coord_near)
            #     pred_near = geo_mlp.sdf(feature_near)
            #     g_near = get_gradient(coord_near, pred_near)*sigma_sigmoid

            cur_loss = 0.
            
            # weight = torch.abs(weight) # weight's sign indicate the sample is around the surface or in the free space
            # TODO 每个采样点的sdf loss，先用ray loss做实验，之后再加进去
            # sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction) 
            # cur_loss += sdf_loss
            cdr_loss = 0.
            # pred维度: (4096*6, 1)      
            # 给sdf值加一个sigmoid就是occupancy(the alpha in volume rendering), 参考decoder.py occupancy()
            sdf_pred_copy = sdf_pred.detach()
            pred_occ = torch.sigmoid(sdf_pred_copy/sigma_size) # as occ. prob.
            # pred_ray维度: (4096, 6)
            pred_ray = pred_occ.reshape(config.bs, -1)
            # sample_depth reshape后维度: (4096, 6)
            sample_depth = sample_depth.reshape(config.bs, -1)
            color_pred = color_pred.reshape(config.bs, -1, 3)
            if config.main_loss_type == "dr":
                # ray_depth维度: (4096, 1)
                cdr_loss = color_depth_rendering_loss(sample_depth, pred_ray, ray_depth, color_pred, color_label, neus_on=False)
            elif config.main_loss_type == "dr_neus":
                cdr_loss = color_depth_rendering_loss(sample_depth, pred_ray, ray_depth, color_pred, color_label, neus_on=True)
            cur_loss += cdr_loss * config.cr_loss_weight

            weight = torch.abs(weight)
            sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction) 
            cur_loss += sdf_loss

            # incremental learning regularization loss 
            sdf_reg_loss = 0.
            color_reg_loss = 0.
            if config.continual_learning_reg:
                sdf_reg_loss = sdf_octree.cal_regularization()    
                color_reg_loss = color_octree.cal_regularization()
                cur_loss += config.lambda_forget * sdf_reg_loss
                cur_loss += config.color_lambda_forget * color_reg_loss

            # optional ekional loss
            eikonal_loss = 0.
            if config.ekional_loss_on: # MSE with regards to 1  
                # eikonal_loss = ((g.norm(2, dim=-1) - 1.0) ** 2).mean() # both the surface and the freespace
                # eikonal_loss = ((g[~surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # only the freespace
                eikonal_loss = ((g[surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # only close to the surface
                cur_loss += config.weight_e * eikonal_loss
            
            consistency_loss = 0.
            if config.consistency_loss_on:
                consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
                cur_loss += config.weight_c * consistency_loss
            

            opt.zero_grad(set_to_none=True)
            cur_loss.backward(retain_graph=True) # this is the slowest part (about 10x the forward time)
            opt.step()

            total_iter += 1

            if config.wandb_vis_on:
                wandb_log_content = {
                    'iter': total_iter, 'loss/total_loss': cur_loss, 
                    'loss/color_depth_rendering_loss': cdr_loss, 'loss/sdf_loss': sdf_loss, \
                    'loss/sdf_reg_loss':sdf_reg_loss, 'loss/color_reg_loss':color_reg_loss, 'loss/eikonal_loss': eikonal_loss, 
                    'loss/consistency_loss': consistency_loss} 
                wandb.log(wandb_log_content)
        
        # calculate the importance of each octree feature
        if config.continual_learning_reg:
            opt.zero_grad(set_to_none=True)
            # TODO 这个计算importance的过程没太看懂，这个是用来更新octree的importance_weight的，这个importance_weight只有在cal_regularization
            # 算regularization loss的时候用到了
            cal_feature_importance(dataset, sdf_octree, color_octree, sdf_mlp, color_mlp, sigma_sigmoid, config.bs, \
                config.cal_importance_weight_down_rate, config.loss_reduction, sigma_size=sigma_size)
            
        print("lidar2camera_matrix's change (percent): ")
        l2c_change = ((lidar2camera_matrix - lidar2camera_matrix_backup) / lidar2camera_matrix_backup) * 100
        l2c_change[3,:3] = lidar2camera_matrix[3, :3] - lidar2camera_matrix_backup[3, :3]
        print(l2c_change)

        print("lidar2camera_matrix:")
        print(lidar2camera_matrix)


        T2 = get_time()
        
        # reconstruction by marching cubes
        if processed_frame == 0 or frame_id == config.end_frame or (processed_frame+1) % config.mesh_freq_frame == 0:
            print("Begin mesh reconstruction from the implicit map")       
            vis_mesh = True 
            # print("Begin reconstruction from implicit mapn")               
            mesh_path = run_path + '/mesh/mesh_frame_' + str(frame_id+1) + ".ply"
            map_path = run_path + '/map/sdf_map_frame_' + str(frame_id+1) + ".ply"
            if config.mc_with_octree: # default
                cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, config.semantic_on)
            else:
                if config.mc_local: # only build the local mesh to speed up
                    cur_mesh = mesher.recon_bbx_mesh(dataset.cur_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                else:
                    cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

        T3 = get_time()

        if config.o3d_vis_on:
            if vis_mesh: 
                cur_mesh.transform(dataset.begin_pose_inv) # back to the globally shifted frame for vis
                vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref, cur_mesh)
            else: # only show frame and current point cloud
                vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref)

            # visualize the octree (it is a bit slow and memory intensive for the visualization)
            # vis_octree = True
            # if vis_octree: 
            #     cur_mesh.transform(dataset.begin_pose_inv)
            #     vis_list = [] # create a list of bbx for the octree nodes
            #     for l in range(config.tree_level_feat):
            #         nodes_coord = octree.get_octree_nodes(config.tree_level_world-l)/config.scale
            #         box_size = np.ones(3) * config.leaf_vox_size * (2**l)
            #         for node_coord in nodes_coord:
            #             node_box = o3d.geometry.AxisAlignedBoundingBox(node_coord-0.5*box_size, node_coord+0.5*box_size)
            #             node_box.color = random_color_table[l]
            #             vis_list.append(node_box)
            #     vis_list.append(cur_mesh)
            #     o3d.visualization.draw_geometries(vis_list)

        if config.wandb_vis_on:
            wandb_log_content = {'frame': processed_frame, 'timing(s)/preprocess': T1-T0, 'timing(s)/mapping': T2-T1, 'timing(s)/reconstruct': T3-T2} 
            wandb.log(wandb_log_content)

        # save checkpoint model
        if ((processed_frame+1) % config.save_freq_frame == 0 or frame_id == config.end_frame) and processed_frame > 0:
            checkpoint_name = 'model/model_frame_' + str(frame_id+1)
            # octree.clear_temp()
            save_checkpoint(sdf_octree, color_octree, sdf_mlp, color_mlp, opt, run_path, checkpoint_name, frame_id + 1, sigma_size)
            save_decoder(sdf_mlp, color_mlp, run_path, checkpoint_name) # save both the gro and sem decoders

        processed_frame += 1
    
    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    run_shine_mapping_incremental()