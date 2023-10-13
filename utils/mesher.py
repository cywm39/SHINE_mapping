import numpy as np
from tqdm import tqdm
import skimage.measure
import torch
import math
import open3d as o3d
import copy
import kaolin as kal
from utils.config import SHINEConfig
from utils.semantic_kitti_utils import *
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from model.color_SDF_decoder import color_SDF_decoder
from model.sdf_decoder import SDFDecoder
from model.color_decoder import ColorDecoder

class Mesher():

    def __init__(self, config: SHINEConfig, sdf_octree: FeatureOctree, color_octree: FeatureOctree, \
        sdf_decoder: SDFDecoder, color_decoder: ColorDecoder, sem_decoder: color_SDF_decoder):

        self.config = config
    
        self.sdf_octree = sdf_octree
        self.color_octree = color_octree
        # self.sdf_color_decoder = sdf_color_decoder
        self.sdf_decoder = sdf_decoder
        self.color_decoder = color_decoder
        self.sem_decoder = sem_decoder
        self.device = config.device
        self.cur_device = self.device
        self.dtype = config.dtype
        self.world_scale = config.scale

        self.ts = 0 # query timestamp when conditioned on time

        self.global_transform = np.eye(4)
    
    def query_points(self, coord, bs, query_sdf = True, query_sem = False, query_mask = True, query_color = False):
        """ query the sdf value, semantic label and marching cubes mask for points
        Args:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            bs: batch size for the inference
        Returns:
            sdf_pred: Ndim numpy array, signed distance value (scaled) at each query point
            sem_pred: Ndim numpy array, semantic label prediction at each query point
            mc_mask:  Ndim bool numpy array, marching cubes mask at each query point
            color_pred: [N, 3]dim numpy array, color at each query point
        """
        # the coord torch tensor is already scaled in the [-1,1] coordinate system
        sample_count = coord.shape[0]
        iter_n = math.ceil(sample_count/bs)
        check_level = min(self.sdf_octree.featured_level_num, self.config.mc_vis_level)-1
        if query_sdf:
            sdf_pred = np.zeros(sample_count)
        else: 
            sdf_pred = None
        if query_sem:
            sem_pred = np.zeros(sample_count)
        else:
            sem_pred = None
        if query_mask:
            mc_mask = np.zeros(sample_count)
        else:
            mc_mask = None
        if query_color:
            color_pred = np.zeros((sample_count, 3))
        else:
            color_pred = None
        
        with torch.no_grad(): # eval step
            if iter_n > 1:
                for n in tqdm(range(iter_n)):
                    head = n*bs
                    tail = min((n+1)*bs, sample_count)
                    batch_coord = coord[head:tail, :]
                    if self.cur_device == "cpu" and self.device == "cuda":
                        batch_coord = batch_coord.cuda()
                    batch_sdf_feature = self.sdf_octree.query_feature(batch_coord, True) # query features
                    batch_color_feature = self.color_octree.query_feature(batch_coord, True) # query features
                    # if query_sdf or query_color:
                    #     if not self.config.time_conditioned:
                    #         batch_sdf, batch_color = self.sdf_color_decoder(batch_sdf_feature, batch_color_feature)
                    #         batch_sdf = -batch_sdf
                    #     else:
                    #         batch_sdf = -self.sdf_color_decoder.time_conditionded_sdf(batch_sdf_feature, self.ts * torch.ones(batch_sdf_feature.shape[0], 1).cuda())
                    #     if query_sdf:
                    #         sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()
                    #     if query_color:
                    #         color_pred[head:tail] = batch_color.detach().cpu().numpy()
                    if query_sdf:
                        if not self.config.time_conditioned:
                            batch_sdf = self.sdf_decoder.predict_sdf(batch_sdf_feature)
                            batch_sdf = -batch_sdf
                        else:
                            batch_sdf = -self.sdf_decoder.time_conditionded_sdf(batch_sdf_feature, self.ts * torch.ones(batch_sdf_feature.shape[0], 1).cuda())
                        sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()
                    if query_color:
                        if not self.config.time_conditioned:
                            batch_color = self.color_decoder.predict_color(batch_color_feature)
                        else:
                            batch_color = self.color_decoder.time_conditionded_color(batch_color_feature, self.ts * torch.ones(batch_color_feature.shape[0], 1).cuda())
                        color_pred[head:tail] = batch_color.detach().cpu().numpy()
                    if query_sem:
                        batch_sem = self.sem_decoder.sem_label(batch_sdf_feature)
                        sem_pred[head:tail] = batch_sem.detach().cpu().numpy()
                    if query_mask:
                        # get the marching cubes mask
                        # hierarchical_indices: bottom-up
                        check_level_indices = self.sdf_octree.hierarchical_indices[check_level] 
                        # print(check_level_indices)
                        # if index is -1 for the level, then means the point is not valid under this level
                        mask_mc = check_level_indices >= 0
                        # print(mask_mc.shape)
                        # all should be true (all the corner should be valid)
                        mask_mc = torch.all(mask_mc, dim=1)
                        mc_mask[head:tail] = mask_mc.detach().cpu().numpy()
                        # but for scimage's marching cubes, the top right corner's mask should also be true to conduct marching cubes
            else:
                sdf_feature = self.sdf_octree.query_feature(coord, True)
                color_feature = self.color_octree.query_feature(coord, True)
                # if query_sdf or query_color:
                #     if not self.config.time_conditioned:
                #         batch_sdf, batch_color = self.sdf_color_decoder(sdf_feature, color_feature)
                #         batch_sdf = -batch_sdf
                #         if query_sdf:
                #             sdf_pred = batch_sdf.detach().cpu().numpy()
                #         if query_color:
                #             color_pred = batch_color.detach().cpu().numpy()
                #     else: # just for a quick test
                #         sdf_pred = -self.sdf_color_decoder.time_conditionded_sdf(sdf_feature, self.ts * torch.ones(sdf_feature.shape[0], 1).cuda()).detach().cpu().numpy()
                if query_sdf:
                    if not self.config.time_conditioned:
                        sdf_pred = self.sdf_decoder.predict_sdf(sdf_feature)
                        sdf_pred = -sdf_pred
                        sdf_pred = sdf_pred.detach().cpu().numpy()
                    else: 
                        sdf_pred = -self.sdf_decoder.time_conditionded_sdf(sdf_feature, self.ts * torch.ones(sdf_feature.shape[0], 1).cuda()).detach().cpu().numpy()
                if query_color:
                    if not self.config.time_conditioned:
                        color_pred = self.color_decoder.predict_color(color_feature)
                        color_pred = color_pred.detach().cpu().numpy()
                    else:
                        color_pred = self.color_decoder.time_conditionded_color(color_feature, self.ts * torch.ones(color_feature.shape[0], 1).cuda()).detach().cpu().numpy()
                if query_sem:
                    sem_pred = self.sem_decoder.sem_label(sdf_feature).detach().cpu().numpy()
                if query_mask:
                    # get the marching cubes mask
                    check_level_indices = self.sdf_octree.hierarchical_indices[check_level] 
                    # if index is -1 for the level, then means the point is not valid under this level
                    mask_mc = check_level_indices >= 0
                    # all should be true (all the corner should be valid)
                    mc_mask = torch.all(mask_mc, dim=1).detach().cpu().numpy()

        return sdf_pred, sem_pred, mc_mask, color_pred

    def get_query_from_bbx(self, bbx, voxel_size):
        """ get grid query points inside a given bounding box (bbx)
        Args:
            bbx: open3d bounding box, in world coordinate system, with unit m 
            voxel_size: scalar, marching cubes voxel size with unit m
        Returns:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            voxel_num_xyz: 3dim numpy array, the number of voxels on each axis for the bbx
            voxel_origin: 3dim numpy array the coordinate of the bottom-left corner of the 3d grids 
                for marching cubes, in world coordinate system with unit m      
        """
        # bbx and voxel_size are all in the world coordinate system
        min_bound = bbx.get_min_bound()
        max_bound = bbx.get_max_bound()
        len_xyz = max_bound - min_bound
        voxel_num_xyz = (np.ceil(len_xyz/voxel_size)+self.config.pad_voxel*2).astype(np.int_)
        voxel_origin = min_bound-self.config.pad_voxel*voxel_size
        # pad an additional voxel underground to gurantee the reconstruction of ground
        voxel_origin[2]-=voxel_size
        voxel_num_xyz[2]+=1

        voxel_count_total = voxel_num_xyz[0] * voxel_num_xyz[1] * voxel_num_xyz[2]
        if voxel_count_total > 5e8: # TODO: avoid gpu memory issue, dirty fix
            self.cur_device = "cpu" # firstly save in cpu memory (which would be larger than gpu's)
            print("too much query points, use cpu memory")
        x = torch.arange(voxel_num_xyz[0], dtype=torch.int16, device=self.cur_device)
        y = torch.arange(voxel_num_xyz[1], dtype=torch.int16, device=self.cur_device)
        z = torch.arange(voxel_num_xyz[2], dtype=torch.int16, device=self.cur_device)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing='ij') 
        # get the vector of all the grid point's 3D coordinates
        coord = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        # transform to world coordinate system
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=self.dtype, device=self.cur_device)
        # scaling to the [-1, 1] coordinate system
        coord *= self.world_scale
        
        return coord, voxel_num_xyz, voxel_origin
    
    def generate_sdf_map(self, coord, sdf_pred, mc_mask, map_path):
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        sdf_map_pc = o3d.t.geometry.PointCloud(device)

        # scaling back to the world coordinate system
        coord /= self.world_scale
        coord_np = coord.detach().cpu().numpy()

        sdf_pred_world = sdf_pred * self.config.logistic_gaussian_ratio*self.config.sigma_sigmoid_m # convert to unit: m

        # the sdf (unit: m) would be saved in the intensity channel
        sdf_map_pc.point['positions'] = o3d.core.Tensor(coord_np, dtype, device)
        sdf_map_pc.point['intensities'] = o3d.core.Tensor(np.expand_dims(sdf_pred_world, axis=1), dtype, device) # scaled sdf prediction
        if mc_mask is not None:
            # the marching cubes mask would be saved in the labels channel (indicating the hierarchical position in the octree)
            sdf_map_pc.point['labels'] = o3d.core.Tensor(np.expand_dims(mc_mask, axis=1), o3d.core.int32, device) # mask

        # global transform (to world coordinate system) before output
        sdf_map_pc.transform(self.global_transform)
        o3d.t.io.write_point_cloud(map_path, sdf_map_pc, print_progress=False)
        print("save the sdf map to %s" % (map_path))
    
    def assign_to_bbx(self, sdf_pred, sem_pred, mc_mask, voxel_num_xyz):
        """ assign the queried sdf, semantic label and marching cubes mask back to the 3D grids in the specified bounding box
        Args:
            sdf_pred: Ndim np.array
            sem_pred: Ndim np.array
            mc_mask:  Ndim bool np.array
            voxel_num_xyz: 3dim numpy array, the number of voxels on each axis for the bbx
        Returns:
            sdf_pred:  a*b*c np.array, 3d grids of sign distance values
            sem_pred:  a*b*c np.array, 3d grids of semantic labels
            mc_mask:   a*b*c np.array, 3d grids of marching cube masks, marching cubes only on where 
                the mask is true
        """
        if sdf_pred is not None:
            sdf_pred = sdf_pred.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2])

        if sem_pred is not None:
            sem_pred = sem_pred.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2])

        if mc_mask is not None:
            mc_mask = mc_mask.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]).astype(dtype=bool)
            # mc_mask[:,:,0:1] = True 

        # if color_pred is not None:
        #     color_pred = color_pred.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2], 3)
            
        return sdf_pred, sem_pred, mc_mask

    def mc_mesh(self, mc_sdf, mc_mask, voxel_size, mc_origin):
        """ use the marching cubes algorithm to get mesh vertices and faces
        Args:
            mc_sdf:  a*b*c np.array, 3d grids of sign distance values
            mc_mask: a*b*c np.array, 3d grids of marching cube masks, marching cubes only on where 
                the mask is true
            voxel_size: scalar, marching cubes voxel size with unit m
            mc_origin: 3*1 np.array, the coordinate of the bottom-left corner of the 3d grids for 
                marching cubes, in world coordinate system with unit m
        Returns:
            ([verts], [faces]), mesh vertices and triangle faces
        """
        print("Marching cubes ...")
        # the input are all already numpy arraies
        verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        try:
            verts, faces, normals, values = skimage.measure.marching_cubes(
                mc_sdf, level=0.0, allow_degenerate=False, mask=mc_mask)
            # # 在获取的顶点上对颜色进行采样
            # colors = mc_color[verts[:, 0], verts[:, 1], verts[:, 2]]
        except:
            pass
        # 只用mc_sdf就能通过调用skimage.measure.marching_cubes得到顶点和faces坐标，考虑到mc_sdf只有每个mc voxel处的sdf值没有坐标信息，
        # 所以这里得到的verts只是相对坐标，需要进行转换
        # 乘以真实世界下的mc voxel边长，再加上真实世界下的origin坐标，得到的就是真实世界下mesh中所有顶点的坐标了
        verts = mc_origin + verts * voxel_size
        return verts, faces

    def estimate_vertices_sem(self, mesh, verts, filter_free_space_vertices = True):
        print("predict semantic labels of the vertices")
        verts_scaled = torch.tensor(verts * self.world_scale, dtype=self.dtype, device=self.device)
        _, verts_sem, _ = self.query_points(verts_scaled, self.config.infer_bs, False, True, False)
        verts_sem_list = list(verts_sem)
        verts_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in verts_sem_list]
        verts_sem_rgb = np.asarray(verts_sem_rgb, dtype=np.float64)/255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(verts_sem_rgb)

        # filter the freespace vertices
        if filter_free_space_vertices:
            non_freespace_idx = verts_sem <= 0
            mesh.remove_vertices_by_mask(non_freespace_idx)
        
        return mesh

    def filter_isolated_vertices(self, mesh, filter_cluster_min_tri = 300):
        # print("Cluster connected triangles")
        triangle_clusters, cluster_n_triangles, _ = (mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        # cluster_area = np.asarray(cluster_area)
        # print("Remove the small clusters")
        # mesh_0 = copy.deepcopy(mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
        mesh.remove_triangles_by_mask(triangles_to_remove)
        # mesh = mesh_0
        return mesh

    def recon_bbx_mesh(self, bbx, voxel_size, mesh_path, map_path, \
        save_map = False, estimate_sem = False, estimate_normal = True, \
        filter_isolated_mesh = True, filter_free_space_vertices = True):
        
        # reconstruct and save the (semantic) mesh from the feature octree the decoders within a
        # given bounding box.
        # bbx and voxel_size all with unit m, in world coordinate system

        coord, voxel_num_xyz, voxel_origin = self.get_query_from_bbx(bbx, voxel_size)
        sdf_pred, _, mc_mask = self.query_points(coord, self.config.infer_bs, True, False, self.config.mc_mask_on)
        if save_map:
            self.generate_sdf_map(coord, sdf_pred, mc_mask, map_path)
        mc_sdf, _, mc_mask = self.assign_to_bbx(sdf_pred, None, mc_mask, voxel_num_xyz)
        verts, faces = self.mc_mesh(mc_sdf, mc_mask, voxel_size, voxel_origin)

        # directly use open3d to get mesh
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.astype(np.float64)),
            o3d.utility.Vector3iVector(faces)
        )

        if estimate_sem: 
            mesh = self.estimate_vertices_sem(mesh, verts, filter_free_space_vertices)

        if estimate_normal:
            mesh.compute_vertex_normals()
        
        if filter_isolated_mesh:
            mesh = self.filter_isolated_vertices(mesh, self.config.min_cluster_vertices)

        # global transform (to world coordinate system) before output
        mesh.transform(self.global_transform)

        # write the mesh to ply file
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("save the mesh to %s\n" % (mesh_path))

        return mesh

    # reconstruct the map sparsely using the octree, only query the sdf at certain level ($query_level) of the octree
    # much faster and also memory-wise more efficient
    def recon_octree_mesh(self, query_level, mc_res_m, mesh_path, map_path, \
                          save_map = False, estimate_sem = False, estimate_normal = True, \
                          filter_isolated_mesh = True, filter_free_space_vertices = True): 

        # query_level层的所有节点坐标，由于是从octree里直接得到所以是[-1,1]坐标系里的
        nodes_coord_scaled = self.sdf_octree.get_octree_nodes(query_level) # query level top-down
        nodes_count = nodes_coord_scaled.shape[0]
        min_nodes = np.min(nodes_coord_scaled, 0) # 最小和最大坐标点
        max_nodes = np.max(nodes_coord_scaled, 0)

        # [-1, 1]坐标系内，query_level层的voxel size(voxel的边长)
        node_res_scaled = 2**(1-query_level) # voxel size for queried octree node in [-1,1] coordinate system
        # marching cube's voxel size should be evenly divisible by the queried octree node's size
        # node_res_scaled / self.world_scale代表实际上的query_level层的voxel size，mc_res_m代表mc算法实际上的voxel size
        # mc算法有一个独立的voxel grid，mc_res_m就是这个voxel grid中每个voxel的边长，是预先设置好的
        # 所以voxel_count_per_side_node就代表八叉树中query_level层的一个voxel在一条边上能放几个mc算法设置的voxel边
        # 由于mc_res_m是config里设置的，显然如果越小，mc算法越细致。
        # 只要理解了mc算法有自己的独立的voxel grid，并且octree的一个voxel里有很多mc算法的voxel，就容易理解这里了
        voxel_count_per_side_node = np.ceil(node_res_scaled / self.world_scale / mc_res_m).astype(dtype=int) 
        # assign coordinates for the queried octree node
        # x y z并不是mc算法在整个空间内要查询的坐标，而只是octree的单个voxel中要查询的坐标
        # 例如如果voxel_count_per_side_node是8，则octree的单个voxel的每一边上，都可以放8个mc算法的voxel
        # 并且这里的坐标是对所有octree的voxel而言的，也就是这里的坐标只是“坐标偏移”，在后面能看到还会加上octree的voxel坐标来计算真实查询坐标
        x = torch.arange(voxel_count_per_side_node, dtype=torch.int16, device=self.device)
        y = torch.arange(voxel_count_per_side_node, dtype=torch.int16, device=self.device)
        z = torch.arange(voxel_count_per_side_node, dtype=torch.int16, device=self.device)
        node_box_size = (np.ones(3)*voxel_count_per_side_node).astype(dtype=int)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing='ij') 
        # get the vector of all the grid point's 3D coordinates
        # 这里的coord就是上面说的，octree的单个voxel里能容纳的所有mc算法的voxel的坐标
        coord = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float() 
        # node_res_scaled是[-1,1]坐标系里octree voxel的边长，voxel_count_per_side_node是octree voxel一条边上能容纳多少mc voxel
        # 所以这里的mc_res_scaled就是[-1,1]坐标系里mc voxel的边长
        mc_res_scaled = node_res_scaled / voxel_count_per_side_node # voxel size for marching cubes in [-1,1] coordinate system
        # transform to [-1,1] coordinate system
        # coord原本是[0,0,0], [0,0,1], [0,0,2]之类的，乘以[-1,1]坐标系下mc voxel的边长得到的就是[-1,1]坐标系下
        # octree的单个voxel中所容纳的所有的mc voxel的坐标
        coord *= mc_res_scaled

        # the voxel count for the whole map
        # voxel_count_per_side就是整个地图中三个方向上有多少mc voxel
        voxel_count_per_side = ((max_nodes - min_nodes)/mc_res_scaled+voxel_count_per_side_node).astype(int)
        # initialize the whole map
        # query_grid_sdf和query_grid_mask用来存放整个地图上所有mc voxel查询得到的值
        query_grid_sdf = np.zeros((voxel_count_per_side[0], voxel_count_per_side[1], voxel_count_per_side[2]), dtype=np.float16) # use float16 to save memory
        query_grid_mask = np.zeros((voxel_count_per_side[0], voxel_count_per_side[1], voxel_count_per_side[2]), dtype=bool)  # mask off
        # query_grid_color = np.zeros((voxel_count_per_side[0], voxel_count_per_side[1], voxel_count_per_side[2], 3), dtype=np.float32)

        # 基本流程是遍历octree的query_level层里所有的voxel，得到这个voxel的origin坐标，和coord相加得到该voxel里所有要查询的mc voxel的坐标
        # 要查询的mc voxel坐标输入网络得到sdf和mask，reshape后就存到query_grid_sdf和query_grid_mask里对应的位置里
        # (因为query_grid_sdf和query_grid_mask里放的是整个地图所有mc voxel的结果，所以当前octree voxel里mc voxel的结果肯定得找到对应位置再赋值)
        for node_idx in tqdm(range(nodes_count)):
            node_coord_scaled = nodes_coord_scaled[node_idx, :]
            cur_origin = torch.tensor(node_coord_scaled - 0.5 * (node_res_scaled - mc_res_scaled), device=self.device)
            cur_coord = coord.clone()
            cur_coord += cur_origin
            cur_sdf_pred, _, cur_mc_mask, cur_color_pred = self.query_points(cur_coord, self.config.infer_bs, True, False, self.config.mc_mask_on, False)
            cur_sdf_pred, _, cur_mc_mask = self.assign_to_bbx(cur_sdf_pred, None, cur_mc_mask, node_box_size)
            shift_coord = (node_coord_scaled - min_nodes)/node_res_scaled
            shift_coord = (shift_coord*voxel_count_per_side_node).astype(int)
            query_grid_sdf[shift_coord[0]:shift_coord[0]+voxel_count_per_side_node, 
                           shift_coord[1]:shift_coord[1]+voxel_count_per_side_node, shift_coord[2]:shift_coord[2]+voxel_count_per_side_node] = cur_sdf_pred
            query_grid_mask[shift_coord[0]:shift_coord[0]+voxel_count_per_side_node, 
                            shift_coord[1]:shift_coord[1]+voxel_count_per_side_node, shift_coord[2]:shift_coord[2]+voxel_count_per_side_node] = cur_mc_mask
            # query_grid_color[shift_coord[0]:shift_coord[0]+voxel_count_per_side_node, 
            #                  shift_coord[1]:shift_coord[1]+voxel_count_per_side_node, shift_coord[2]:shift_coord[2]+voxel_count_per_side_node, :] = cur_color_pred
        
        # mc_res_scaled是[-1, 1]坐标系中mc voxel的边长，除以world_scale就是真实世界下mc voxel的边长
        # 其实这一步没用，因为mc_voxel_size就等于mc_res_m
        mc_voxel_size = mc_res_scaled / self.world_scale
        # mc_voxel_origin就是真实世界下，min_nodes的origin坐标
        mc_voxel_origin = (min_nodes - 0.5 * (node_res_scaled - mc_res_scaled)) / self.world_scale

        # if save_map: # ignore it now, too much for the memory
        #     # query_grid_coord 
        #     self.generate_sdf_map(query_grid_coord, query_grid_sdf, query_grid_mask, map_path)

        # 注意这里的verts是真实世界下mesh中所有顶点坐标
        verts, faces = self.mc_mesh(query_grid_sdf, query_grid_mask, mc_voxel_size, mc_voxel_origin)
        # directly use open3d to get mesh
        
        points = torch.tensor(verts, device=self.device)
        # octree的查询只支持[-1,1]坐标系下，所以需要对真实世界下的顶点坐标进行缩放，乘以world_scale
        points *= self.world_scale
        sdf_pred, _, mc_mask, color_pred = self.query_points(points, self.config.infer_bs, False, False, False, True)
        vertex_colors = color_pred
        vertex_colors = np.clip(vertex_colors, 0, 255)
        vertex_colors /= 255.0

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces)
        )
        # 得到的颜色本身和坐标没啥关系，只是按照坐标的顺序排列的，所以可以直接给mesh赋值
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        # if colors is not None:
        #     # 设置顶点颜色
        #     mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        if estimate_sem: 
            mesh = self.estimate_vertices_sem(mesh, verts, filter_free_space_vertices)

        if estimate_normal:
            mesh.compute_vertex_normals()
        
        if filter_isolated_mesh:
            mesh = self.filter_isolated_vertices(mesh)

        # global transform (to world coordinate system) before output
        mesh.transform(self.global_transform)

        # write the mesh to ply file
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("save the mesh to %s\n" % (mesh_path))

        return mesh