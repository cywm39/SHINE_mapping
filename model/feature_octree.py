import torch
import torch.nn as nn

import time
from tqdm import tqdm
import kaolin as kal
import numpy as np

from functools import partial
from collections import defaultdict

import multiprocessing

from utils.config import SHINEConfig


def get_dict_values(dictionary, keys, default=None):
    return [dictionary.get(key, default) for key in keys]

def parallel_get_dict_values(dictionary, key_list, default=None, processes=None):
    if processes is None:
        processes = multiprocessing.cpu_count()
    chunk_size = len(key_list) // processes
    chunks = [key_list[i:i+chunk_size] for i in range(0, len(key_list), chunk_size)]
    with multiprocessing.Pool(processes=processes) as pool:
        value_list = pool.starmap(get_dict_values, [(dictionary, keys, default) for keys in chunks])
    return [v for values in value_list for v in values]

class FeatureOctree(nn.Module):

    def __init__(self, config: SHINEConfig, is_color: bool):
        
        super().__init__()

        if is_color:
            # [0 1 2 3 ... max_level-1 max_level], 0 level is the root, which have 8 corners.
            self.max_level = config.color_tree_level_world
            # the number of levels with feature (begin from bottom)
            self.leaf_vox_size = config.color_leaf_vox_size 
            self.featured_level_num = config.color_tree_level_feat 
            self.free_level_num = self.max_level - self.featured_level_num + 1
            self.feature_dim = config.color_feature_dim
            self.feature_std = config.color_feature_std
            self.polynomial_interpolation = config.color_poly_int_on
        else:        
            # [0 1 2 3 ... max_level-1 max_level], 0 level is the root, which have 8 corners.
            self.max_level = config.tree_level_world
            # the number of levels with feature (begin from bottom)
            self.leaf_vox_size = config.leaf_vox_size 
            self.featured_level_num = config.tree_level_feat 
            self.free_level_num = self.max_level - self.featured_level_num + 1
            self.feature_dim = config.feature_dim
            self.feature_std = config.feature_std
            self.polynomial_interpolation = config.poly_int_on
            
        self.device = config.device
        self.is_color = is_color

        # Initialize the look up tables 
        self.corners_lookup_tables = [] # from corner morton to corner index (top-down)
        self.nodes_lookup_tables = []   # from nodes morton to corner index (top-down)
        # Initialize the look up table for each level, each is a dictionary
        # 为什么是max_level+1？
        for l in range(self.max_level+1):
            self.corners_lookup_tables.append({})
            self.nodes_lookup_tables.append({}) # actually the same speed as below
            # default_indices = [-1 for i in range(8)]
            # nodes_dict = defaultdict(lambda:default_indices)
            # self.nodes_lookup_tables.append(nodes_dict)
            
        # Initialize the hierarchical grid feature list 
        if self.featured_level_num < 1:
            raise ValueError('No level with grid features!')
        # hierarchical grid features list
        # top-down: leaf node level is stored in the last row (dim feature_level_num-1)
        # but only for the featured levels
        self.hier_features = nn.ParameterList([]) 

        # the temporal stuffs that can be cleared
        # hierachical feature grid indices for the input batch point
        self.hierarchical_indices = [] 
        # bottom-up: stored from the leaf node level (dim 0)

        # used for incremental learning (mapping)
        self.importance_weight = [] # weight for each feature dimension
        self.features_last_frame = [] # hierarchical features for the last frame

        self.to(config.device)

    # the last element of the each level of the hier_features is the trashbin element
    # after the optimization, we need to set it back to zero vector
    def set_zero(self):
        with torch.no_grad():
            for n in range(len(self.hier_features)):
                self.hier_features[n][-1] = torch.zeros(1,self.feature_dim) 

    def forward(self, x):
        feature = self.query_feature(x)
        return feature

    def get_morton(self, sample_points, level):
        # 这个坐标转换函数将[-1,1]的点转换到整数坐标，可能unbatched_pointcloud_to_spc得到的spc里面所有坐标也是这么转换得到的整数坐标？
        points = kal.ops.spc.quantize_points(sample_points, level)  # quantize to interger coords
        points_morton = kal.ops.spc.points_to_morton(points) # to 1d morton code
        sample_points_with_morton = torch.hstack((sample_points, points_morton.view(-1, 1)))
        morton_set = set(points_morton.cpu().numpy())
        return sample_points_with_morton, morton_set

    def get_octree_nodes(self, level): # top-down
        nodes_morton = list(self.nodes_lookup_tables[level].keys())
        nodes_morton = torch.tensor(nodes_morton).to(self.device, torch.int64)
        nodes_spc = kal.ops.spc.morton_to_points(nodes_morton)
        nodes_spc_np = nodes_spc.cpu().numpy()
        node_size = 2**(1-level) # in the -1 to 1 kaolin space
        # 假设unbatched_pointcloud_to_spc函数得到的spc会类似于上面的quantize_points函数一样，将原来[-1,1]的点的坐标区间变成一个整数坐标
        # 那这里由于是逆过程(从spc中点的坐标得到原始点的坐标)，所以这里的目的就是从整数坐标区间变成[-1,1]坐标区间
        # 也即，原始点坐标是[-1,1]区间，octree或者说spc内的点的坐标是整数区间，unbatched_pointcloud_to_spc和这里都是在做变换
        nodes_coord_scaled = (nodes_spc_np * node_size) - 1. + 0.5 * node_size  # in the -1 to 1 kaolin space
        return nodes_coord_scaled

    def is_empty(self):
        return len(self.hier_features) == 0

    # clear the temp data (used for one batch) that is not needed
    def clear_temp(self):
        self.hierarchical_indices = [] 
        self.importance_weight = []
        self.features_last_frame = []

    # update the octree according to new observations
    # if incremental_on = True, then we additional store the last frames' feature for regularization based incremental mapping
    # update只在lidar_dataset.py中process_frame的时候被调用了，传递的surface_points都是scale后的坐标，也即[-1,1]空间内的坐标
    def update(self, surface_points, incremental_on = False):
        # [0 1 2 3 ... max_level-1 max_level]
        spc = kal.ops.conversions.unbatched_pointcloud_to_spc(surface_points, self.max_level) 
        pyramid = spc.pyramids[0].cpu()
        for i in range(self.max_level+1): # for each level (top-down)   
            # 为什么没有feature的层不用更新？因为update函数要更新的就是两个lookup_tables和hier_features，
            # 实际上这个FeatureOctree类根本没存储整个octree，只存了feature和两个lookup_tables，
            # lidar扫描到的新的点直接通过kaolin库的unbatched_pointcloud_to_spc方法计算得到spc，从而转变为octree结构
            if i < self.free_level_num: # free levels (skip), only need to consider the featured levels
                continue
            # level storing features (i>=free_level_num)
            nodes = spc.point_hierarchies[pyramid[1, i]:pyramid[1, i+1]]
            nodes_morton = kal.ops.spc.points_to_morton(nodes).cpu().numpy().tolist() # nodes at certain level
            new_nodes_index = []
            for idx in range(len(nodes_morton)):
                if nodes_morton[idx] not in self.nodes_lookup_tables[i]:
                    new_nodes_index.append(idx) # nodes to corner dictionary: key is the morton code
            new_nodes = nodes[new_nodes_index] # get the newly added nodes
            if new_nodes.shape[0] == 0:
                continue
            corners = kal.ops.spc.points_to_corners(new_nodes).reshape(-1,3) 
            corners_unique = torch.unique(corners, dim=0)
            # mortons of the coners from the new scan
            corners_morton = kal.ops.spc.points_to_morton(corners_unique).cpu().numpy().tolist()
            if len(self.corners_lookup_tables[i]) == 0: # for the first frame
                corners_dict = dict(zip(corners_morton, range(len(corners_morton))))
                self.corners_lookup_tables[i] = corners_dict
                # initializa corner features
                fts = self.feature_std*torch.randn(len(corners_dict)+1, self.feature_dim, device=self.device) 
                # 空出来的这最后一个是干啥的，下面那个weights也多出来一个
                fts[-1] = torch.zeros(1,self.feature_dim)
                # Be careful, the size of the feature list equals to featured_level_num not max_level+1
                self.hier_features.append(nn.Parameter(fts)) 
                if incremental_on:
                    weights = torch.zeros(len(corners_dict)+1, self.feature_dim, device=self.device) 
                    self.importance_weight.append(weights)
                    self.features_last_frame.append(fts.clone())
            else: # update for new frames
                pre_size = len(self.corners_lookup_tables[i])
                for m in corners_morton:
                    if m not in self.corners_lookup_tables[i]: # add new keys
                        self.corners_lookup_tables[i][m] = len(self.corners_lookup_tables[i])
                new_feature_num = len(self.corners_lookup_tables[i]) - pre_size
                new_fts = self.feature_std*torch.randn(new_feature_num+1, self.feature_dim, device=self.device) 
                new_fts[-1] = torch.zeros(1,self.feature_dim)
                cur_featured_level = i-self.free_level_num
                # hier_features中的每一项都是个list，表示某一层的所有feature向量
                # 每一层的feature向量的个数是corners点的个数，而不是扫描到的点(nodes)的个数，扫描到的点用来计算这些corners点
                # 从上面的len(corners_dict)+1可以看到每一层的最后一个元素是额外多出来的，从set_zero函数可以看到最后这个元素是trashbin元素
                # 虽然没搞清楚trashbin元素是干啥的，不过优化完毕需要调用set_zero函数把这个trashbin元素设置成零向量
                self.hier_features[cur_featured_level] = nn.Parameter(torch.cat((self.hier_features[cur_featured_level][:-1],new_fts),0))
                if incremental_on:
                    new_weights = torch.zeros(new_feature_num+1, self.feature_dim, device=self.device)
                    self.importance_weight[cur_featured_level] = torch.cat((self.importance_weight[cur_featured_level][:-1],new_weights),0)
                    self.features_last_frame[cur_featured_level] = (self.hier_features[cur_featured_level].clone())

            corners_m = kal.ops.spc.points_to_morton(corners).cpu().numpy().tolist()
            indexes = torch.tensor([self.corners_lookup_tables[i][x] for x in corners_m]).reshape(-1,8).numpy().tolist()
            new_nodes_morton = kal.ops.spc.points_to_morton(new_nodes).cpu().numpy().tolist()
            for k in range(len(new_nodes_morton)):
                # 从这里看出来，nodes_lookup_tables[i]这个字典，key是nodes的morton码，value是对应的node求出的**八**个corner点的index。
                #   同时，key包含的其实就是这第i层的所有的nodes: 输入的新扫描的一帧点云，通过kaolin库函数直接计算得到分层的spc，其实就是octree,
                #   之后遍历spc的每一层，找出每一层中所有nodes里新添加的nodes，并插入到nodes_lookup_tables里面
                # 而corners_lookup_tables[i]这个字典，key是所有corner点的morton码，value是该corner点的index
                # index实际上就是对这第i层的所有corner点的一个编码，也即：0 1 2 3 4 5 6 ......
                # 这个index会用来查询hier_features里的feature
                self.nodes_lookup_tables[i][new_nodes_morton[k]] = indexes[k]

        # nodes_coord = self.get_octree_nodes(self.max_level)
        # print(nodes_coord)
        
    # tri-linear (or polynomial) interplation of feature at certain octree level at certain spatial point x 
    def interpolat(self, x, level, polynomial_on = True):
        coords = ((2**level)*(x*0.5+0.5)) 
        d_coords = torch.frac(coords)
        if polynomial_on:
            tx = 3*(d_coords[:,0]**2) - 2*(d_coords[:,0]**3)
            ty = 3*(d_coords[:,1]**2) - 2*(d_coords[:,1]**3)
            tz = 3*(d_coords[:,2]**2) - 2*(d_coords[:,2]**3)
        else: # linear 
            tx = d_coords[:,0]
            ty = d_coords[:,1]
            tz = d_coords[:,2]
        _1_tx = 1-tx
        _1_ty = 1-ty
        _1_tz = 1-tz
        p0 = _1_tx*_1_ty*_1_tz
        p1 = _1_tx*_1_ty*tz
        p2 = _1_tx*ty*_1_tz
        p3 = _1_tx*ty*tz
        p4 = tx*_1_ty*_1_tz
        p5 = tx*_1_ty*tz
        p6 = tx*ty*_1_tz
        p7 = tx*ty*tz

        p = torch.stack((p0,p1,p2,p3,p4,p5,p6,p7),0).T.unsqueeze(2)
        return p

    # get the unique indices of the feature node at spatial points x for each level
    def get_indices(self, coord):
        self.hierarchical_indices = [] # initialize the hierarchical indices list for the batch points x
        for i in range(self.featured_level_num): # bottom-up, for each level
            current_level = self.max_level - i
            points = kal.ops.spc.quantize_points(coord,current_level) # quantize to interger coords
            points_morton = kal.ops.spc.points_to_morton(points).cpu().numpy().tolist() # convert to 1d morton code for the voxel center
            features_last_row = [-1 for t in range(8)] # if not in the look up table, then assign all -1
            # look up the 8 corner nodes' unique indices for each 1d morton code in the look up table [nx8], the most time-consuming part 
            # [actually a kind of hashing realized by python dictionary]
            
            indices_list = [self.nodes_lookup_tables[current_level].get(p,features_last_row) for p in points_morton] 
            # indices_list = [self.nodes_lookup_tables[current_level][p] for p in points_morton]  # actually the same speed as above when using the defaultdict
            # indices_list = parallel_get_dict_values(self.nodes_lookup_tables[current_level], points_morton, features_last_row, 4) # can be sloved by paralle hashing without conflict (no, it's even slower)
            
            # if p is not found in the key lists of cur_lookup_table, use features_last_row, 
            # which is the all-zero trashbin vector of the level's feature
            indices_torch = torch.tensor(indices_list, device=self.device) 
            self.hierarchical_indices.append(indices_torch) # l level {nx8}  # bottom-up
        
        return self.hierarchical_indices


    # get the hierachical-sumed interpolated feature at spatial points x
    def query_feature_with_indices(self, coord, hierarchical_indices):
        # TODO: filter the -1 idx for faster back-propgation
        sum_features = torch.zeros(coord.shape[0], self.feature_dim, device=self.device)
        for i in range(self.featured_level_num): # for each level
            current_level = self.max_level - i
            feature_level = self.featured_level_num-i-1
            # Interpolating
            # get the interpolation coefficients for the 8 neighboring corners, corresponding to the order of the hierarchical_indices
            coeffs = self.interpolat(coord,current_level,self.polynomial_interpolation) 
            # coeffs决定了八个corners点对插值结果的影响，也就是个权重
            # 给出一个点的坐标，遍历所有的level。对于当前level，查询当前点坐标的八个corner点，得到八个feature向量，将八个feature向量根据coeffs加权相加，
            # 得到该点在当前level的插值feature结果。并且各个level的插值feature结果都会相加，level之间权重为1
            # （下面的sum_features += 就实现了level之间相加的步骤，并且因为没有额外权重所以各level之间feature的权重都是1）
            # 最后得到的sum_features是n*8的，n就是coord.shape[0]也就是要查询的点的个数，8就是feature维度（参数里设置的默认值是8）
            # 所以在整个“给定点坐标，在octree中查询feature”过程中，这一步就完成了将octree的各个level的feature相加的步骤，而不是论文里图画的那样在输入网络之前才把各个level之间的feature相加
            # TODO hier_features是单纯一个list，查询方式也就是下标直接查询，那这里好像根本没涉及到hash加速？为什么论文里说用了hash?
            sum_features += (self.hier_features[feature_level][hierarchical_indices[i]]*coeffs).sum(1) 
            # corner index -1 means the queried voxel is not in the leaf node. If so, we will get the trashbin row of the feature grid, 
            # and get the value 0, the feature for this level will then be 0
        return sum_features

    # all-in-one function to get the octree features for a batch of points
    def query_feature(self, coord, faster = False):
        self.set_zero() # set the trashbin feature vector back to 0 after the feature update
        if faster:
            indices = self.get_indices_fast(coord) # it would only be faster when the input coords have a lot share the same morton code
        else:
            indices = self.get_indices(coord)
        features = self.query_feature_with_indices(coord, indices)
        return features
    
    def cal_regularization(self):
        regularization = 0.
        for i in range(self.featured_level_num): # for each level
            feature_level = self.featured_level_num-i-1
            unique_indices = self.hierarchical_indices[i].flatten().unique()
            # feature change between current and last frame
            difference = self.hier_features[feature_level][unique_indices] - self.features_last_frame[feature_level][unique_indices] 
            # regularization for continous learning weighted by the feature importance and the change magnitude    
            regularization += (self.importance_weight[feature_level][unique_indices]*(difference**2)).sum()
        return regularization

    def list_duplicates(self, seq):
        dd = defaultdict(list)
        for i,item in enumerate(seq):
            dd[item].append(i)
        return [(key,locs) for key,locs in dd.items() if len(locs)>=1] 
                                
    # speed up for the batch sdf inferencing during meshing
    # points in the same voxel would be grouped and getting indices together
    # more efficient only when there are lots of samples from the same voxel in the batch (the case when conducting meshing)
    # This function contains some problem which would make the mesh worse, check it later (solved)
    def get_indices_fast(self, coord):
        self.hierarchical_indices = []
        for i in range(self.featured_level_num): # bottom-up
            current_level = self.max_level - i
            points = kal.ops.spc.quantize_points(coord,current_level) # quantize to interger coords
            points_morton = kal.ops.spc.points_to_morton(points).cpu().numpy().tolist() # convert to 1d morton code for the voxel center
            features_last_row = [-1 for t in range(8)] # if not in the look up table, then assign -1

            dups_in_mortons = dict(self.list_duplicates(points_morton)) # list the x with the same morton code (samples inside the same voxel)
            dups_indices = np.zeros((len(points_morton), 8))
            # print(len(dups_in_mortons.keys()), len(points_morton))
            for p in dups_in_mortons.keys():
                idx = dups_in_mortons[p] # indices, p is the point morton
                # get indices only once for these samples sharing the same voxel 
                corner_indices = self.nodes_lookup_tables[current_level].get(p,features_last_row)
                dups_indices[idx,:] = corner_indices
            indices = torch.tensor(dups_indices, device=self.device).long()
            self.hierarchical_indices.append(indices) # l level {nx8} 
        
        return self.hierarchical_indices

    def print_detail(self):
        if self.is_color:
            print("Current Color Octree:")
        else:
            print("Current SDF Octree:")
        total_vox_count = 0
        for level in range(self.featured_level_num):
            level_vox_size = self.leaf_vox_size*(2**(self.featured_level_num-1-level))
            level_vox_count = self.hier_features[level].shape[0]
            print("%.2f m: %d voxel corners" %(level_vox_size, level_vox_count))
            total_vox_count += level_vox_count
        total_map_memory = total_vox_count * self.feature_dim * 4 / 1024 / 1024 # unit: MB
        print("memory: %d x %d x 4 = %.3f MB" %(total_vox_count, self.feature_dim, total_map_memory)) 
        print("--------------------------------")  