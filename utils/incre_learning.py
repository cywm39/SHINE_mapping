import math
from tqdm import tqdm
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from model.color_SDF_decoder import color_SDF_decoder
from model.sdf_decoder import SDFDecoder
from model.color_decoder import ColorDecoder
from dataset.input_dataset import InputDataset
from utils.loss import *

def cal_feature_importance(data: InputDataset, sdf_octree: FeatureOctree, color_octree: FeatureOctree, sdf_mlp: SDFDecoder, 
    color_mlp: ColorDecoder, sigma, bs, down_rate=1, loss_reduction='mean', loss_weight_on = False, sigma_size = None):
    
    # shuffle_indice = torch.randperm(data.coord_pool.shape[0])
    # shuffle_coord = data.coord_pool[shuffle_indice]
    # shuffle_label = data.sdf_label_pool[shuffle_indice]

    sample_count = data.ray_depth_pool.shape[0]
    batch_interval = bs*down_rate
    iter_n = math.ceil(sample_count/batch_interval)

    for n in tqdm(range(iter_n)):
        head = n*batch_interval
        tail = min((n+1)*batch_interval, sample_count)
        # batch_coord = data.coord_pool[head:tail:down_rate]
        # batch_label = data.sdf_label_pool[head:tail:down_rate]
        ray_index = torch.arange(head, tail, down_rate, device=data.pool_device)
        ray_index_repeat = (ray_index * data.ray_sample_count).repeat(data.ray_sample_count, 1)
        sample_index = ray_index_repeat + torch.arange(0, data.ray_sample_count,\
                dtype=int, device=data.device).reshape(-1, 1)
        index = sample_index.transpose(0,1).reshape(-1)
        
        batch_coord = data.coord_pool[index, :].to(data.device)
        batch_label = data.sdf_label_pool[index].to(data.device)
        sample_depth = data.sample_depth_pool[index].to(data.device)
        ray_depth = data.ray_depth_pool[ray_index].to(data.device)
        color_label = data.color_label_pool[ray_index].to(data.device)

        # batch_weight = data.weight_pool[head:tail:down_rate]
        count = batch_label.shape[0]
            
        total_loss = 0.
        sdf_octree.get_indices(batch_coord)
        color_octree.get_indices(batch_coord)
        sdf_features = sdf_octree.query_feature(batch_coord)
        color_features = color_octree.query_feature(batch_coord)
        # sdf_pred, color_pred = mlp(sdf_features, color_features) # before sigmoid         
        sdf_pred = sdf_mlp.predict_sdf(sdf_features)
        color_pred = color_mlp.predict_color(color_features)
        # add options for other losses here                              
        sdf_loss = sdf_bce_loss(sdf_pred, batch_label, sigma, None, loss_weight_on, loss_reduction)
        total_loss += sdf_loss

        pred_occ = torch.sigmoid(sdf_pred/sigma_size) # as occ. prob.
        # pred_ray维度: (4096, 6)
        pred_ray = pred_occ.reshape(ray_index.shape[0], -1)
        # sample_depth reshape后维度: (4096, 6)
        sample_depth = sample_depth.reshape(ray_index.shape[0], -1)
        color_pred = color_pred.reshape(ray_index.shape[0], -1, 3)
        cdr_loss = color_depth_rendering_loss(sample_depth, pred_ray, ray_depth, color_pred, color_label, neus_on=False)
        total_loss += cdr_loss

        total_loss.backward()

        for i in range(len(sdf_octree.importance_weight)): # for each level
            sdf_octree.importance_weight[i] += sdf_octree.hier_features[i].grad.abs()
            sdf_octree.hier_features[i].grad.zero_()
        
            sdf_octree.importance_weight[i][-1] *= 0 # reseting the trashbin feature weight to 0 

        for i in range(len(color_octree.importance_weight)): # for each level
            color_octree.importance_weight[i] += color_octree.hier_features[i].grad.abs()
            color_octree.hier_features[i].grad.zero_()
        
            color_octree.importance_weight[i][-1] *= 0 # reseting the trashbin feature weight to 0 