import numpy as np
from numpy.linalg import inv, norm
from scipy.fftpack import shift
import kaolin as kal
import torch

from utils.config import SHINEConfig

class dataSampler():

    def __init__(self, config: SHINEConfig):

        self.config = config
        self.dev = config.device


    # input and output are all torch tensors
    def sample(self, points_torch, 
               sensor_origin_torch,
               normal_torch,
               sem_label_torch):

        dev = self.dev

        world_scale = self.config.scale
        surface_sample_range_scaled = self.config.surface_sample_range_m * self.config.scale
        surface_sample_n = self.config.surface_sample_n
        clearance_sample_n = self.config.clearance_sample_n # new part

        freespace_sample_n = self.config.free_sample_n
        all_sample_n = surface_sample_n+clearance_sample_n+freespace_sample_n
        free_min_ratio = self.config.free_sample_begin_ratio
        free_sample_end_dist_m_scaled = self.config.free_sample_end_dist_m * self.config.scale
        clearance_dist_scaled = self.config.clearance_dist_m * self.config.scale
        
        sigma_base = self.config.sigma_sigmoid_m * self.config.scale
        # sigma_scale_constant = self.config.sigma_scale_constant

        # get sample points
        # 减去sensor_origin_torch也就是pose中的平移部分是为了把点云移动到原点附近，方便计算点云中点到雷达的距离(直接算点到原点距离就行)
        sensor_origin_torch_tmp = sensor_origin_torch.detach()
        shift_points = points_torch - sensor_origin_torch_tmp
        point_num = shift_points.shape[0]
        # shift_points_tmp = shift_points.detach()
        distances = torch.linalg.norm(shift_points, dim=1, keepdim=True) # ray distances (scaled), 实际上这里的distances就是点到lidar的距离(点的坐标都已经归一化到[-1,1]区间内)
        
        # Part 1. close-to-surface uniform sampling 
        # uniform sample in the close-to-surface range (+- range)
        # (torch.rand(point_num*surface_sample_n, 1, device=dev)-0.5)*2是在[-1,1)范围内随机采样，乘以scale后的所谓“物体表面”的规定距离
        # 就得到了对“物体表面”也就是点云中点的附近进行的采样
        # 注意得到的surface_sample_displacement是在scale后的坐标系下点云中点左右的一个相对值
        # "displacement"就是偏移
        # surface_sample_n是3，所以这里会得到3n个采样点，这些采样点表示对n个点云点的"surface"附近进行的采样，
        # 但采样直接就是得到一个随机数向量，不涉及点云点的坐标，所以这3n个采样点和点云点的对应关系实际上是任意的。当然，一般直接按照shift_points中的点的顺序去对应就行
        surface_sample_displacement = (torch.rand(point_num*surface_sample_n, 1, device=dev)-0.5)*2*surface_sample_range_scaled 
        
        # repeated_dist维度:n*surface_sample_n(假设shift_points总共n个点)
        repeated_dist = distances.repeat(surface_sample_n,1)
        # 这个ratio，以及之后的ratio，含义都能看成是：采样点到雷达的距离 / 点到雷达的距离，分母的那个“点”是指点云中的点，也就是当前采样点对应的那个点云中的点
        # 注意surface_sample_displacement取值范围，其可以是负的也能是正的，可以看成负值表示采样点在物体外，也就是在雷达和点的连线上；正值在物体内，也就是雷达和点连线的延长线上
        surface_sample_dist_ratio = surface_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface
        if sem_label_torch is not None:
            surface_sem_label_tensor = sem_label_torch.repeat(1, surface_sample_n).transpose(0,1)
        
        # Part 2. near surface uniform sampling (for clearance) [from the close surface lower bound closer to the sensor for a clearance distance]
        # part2 这个没太看懂，不过这个设置的clearance_sample_n是0，论文里也没提到这部分的采样点
        clearance_sample_displacement = -torch.rand(point_num*clearance_sample_n, 1, device=dev)*clearance_dist_scaled - surface_sample_range_scaled

        repeated_dist = distances.repeat(clearance_sample_n,1)
        clearance_sample_dist_ratio = clearance_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface
        if sem_label_torch is not None:
            clearance_sem_label_tensor = torch.zeros_like(repeated_dist)

        # Part 3. free space uniform sampling
        repeated_dist = distances.repeat(freespace_sample_n,1)
        free_max_ratio = free_sample_end_dist_m_scaled / repeated_dist + 1.0
        free_diff_ratio = free_max_ratio - free_min_ratio

        # 产生的随机数范围是[free_min_ratio, free_max_ratio)
        # 注意repeat()第二个参数是1的时候不是表示在列上重复，而是让列不重复，也就是只让行重复，所以这里的free_diff_ratio才能和前面维度是(n*3,1)的张量相乘
        free_sample_dist_ratio = torch.rand(point_num*freespace_sample_n, 1, device=dev)*free_diff_ratio + free_min_ratio
        # displacement取值范围是[-0.7repeated_dist,free_sample_end_dist_m_scaled)
        free_sample_displacement = (free_sample_dist_ratio - 1.0) * repeated_dist
        if sem_label_torch is not None:
            free_sem_label_tensor = torch.zeros_like(repeated_dist)
        
        # all together
        all_sample_displacement = torch.cat((surface_sample_displacement, clearance_sample_displacement, free_sample_displacement),0)
        # 注意ratio的含义是采样点和雷达距离 / 点云中对应点到雷达的距离
        all_sample_dist_ratio = torch.cat((surface_sample_dist_ratio, clearance_sample_dist_ratio, free_sample_dist_ratio),0)
        
        repeated_points = shift_points.repeat(all_sample_n,1)
        repeated_dist = distances.repeat(all_sample_n,1)
        # 所有采样点的坐标，由于重新加上了sensor_origin_torch，所以得到的是在scale后世界坐标系下的带有平移和旋转的坐标
        all_sample_points = repeated_points * all_sample_dist_ratio + sensor_origin_torch

        # depth tensor of all the samples
        depths_tensor = repeated_dist * all_sample_dist_ratio
        # 因为除以了world_scale，得到的depths_tensor是采样点到雷达的真实距离，而不是scale后的距离
        depths_tensor /= world_scale # unit: m

        # linear error model: sigma(d) = sigma_base + d * sigma_scale_constant
        # ray_sigma = sigma_base + distances * sigma_scale_constant  
        # different sigma value for different ray with different distance (deprecated)
        # sigma_tensor = ray_sigma.repeat(all_sample_n,1).squeeze(1)

        # get the weight vector as the inverse of sigma 维度(6*n,1)
        weight_tensor = torch.ones_like(depths_tensor)

        # behind surface weight drop-off because we have less uncertainty behind the surface
        if self.config.behind_dropoff_on:
            dropoff_min = self.config.dropoff_min_sigma
            dropoff_max = self.config.dropoff_max_sigma
            dropoff_diff = dropoff_max - dropoff_min
            behind_displacement = (repeated_dist*(all_sample_dist_ratio-1.0)/sigma_base).squeeze(1)
            dropoff_weight = (dropoff_max - behind_displacement) / dropoff_diff
            dropoff_weight = torch.clamp(dropoff_weight, min = 0.0, max = 1.0)
            weight_tensor *= dropoff_weight
        
        # give a flag indicating the type of the sample [negative: freespace, positive: surface]
        # sample点实际上有三种类型，但是这里只规定了surface和freespace的标记
        # 上面torch.cat ratio的时候，前半部分是surface ratio，后半部分是free ratio，所以这里3n之后的全设置成负数
        weight_tensor[point_num*surface_sample_n:] *= -1.0 
        
        # ray-wise depth
        # distances原来是scale后的点云点到雷达的距离，这里除以scale，得到的就是真实的点云点到雷达的距离，类似于上面的depths_tensor
        distances /= world_scale # unit: m
        distances = distances.squeeze(1)

        # assign sdf labels to the samples
        # projective distance as the label: behind +, in-front - 
        # displacement也就是偏移，可以当作采样点处sdf的真值
        sdf_label_tensor = all_sample_displacement.squeeze(1)  # scaled [-1, 1] # as distance (before sigmoid)

        # assign the normal label to the samples
        normal_label_tensor = None
        if normal_torch is not None:
            normal_label_tensor = normal_torch.repeat(all_sample_n,1)
        
        # assign the semantic label to the samples (including free space as the 0 label)
        sem_label_tensor = None
        if sem_label_torch is not None:
            sem_label_tensor = torch.cat((surface_sem_label_tensor, clearance_sem_label_tensor, free_sem_label_tensor),0).int()

        # Convert from the all ray surface + all ray free order to the 
        # ray-wise (surface + free) order
        # all_sample_points维度是(6*n,3)，前半部分是surface，后半部分是free，这里的操作确实会将surface和free的部分分别三个三个组合在一起，
        # 需要注意surface和free的部分的随机点产生是和原来的点云完全没关系的，所以surface部分的采样点和点云点之间没有必要对应关系，可以任意组合
        # 在这个条件下，这里是从surface部分三个三个挑出来，而不是前三个单独拿出来，给all_sample_points举个维度是(12,3)的例子就明白了
        # 总之最后得到的all_sample_points会变成ray-wise的顺序，也就是三个surface采样点+三个free采样点，以此类推，每六个点对应一个点云点的所有采样点
        all_sample_points = all_sample_points.reshape(all_sample_n, -1, 3).transpose(0, 1).reshape(-1, 3)
        sdf_label_tensor = sdf_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1) 
        
        weight_tensor = weight_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        depths_tensor = depths_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)

        if normal_torch is not None:
            normal_label_tensor = normal_label_tensor.reshape(all_sample_n, -1, 3).transpose(0, 1).reshape(-1, 3)
        if sem_label_torch is not None:
            sem_label_tensor = sem_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)

        # ray distance (distances) is not repeated

        return all_sample_points, sdf_label_tensor, normal_label_tensor, sem_label_tensor, \
            weight_tensor, depths_tensor, distances
    

    # space carving sampling (deprecated, to polish)
    def sapce_carving_sample(self, 
                             points_torch, 
                             sensor_origin_torch,
                             space_carving_level,
                             stop_depth_thre,
                             inter_dist_thre):
        
        shift_points = points_torch - sensor_origin_torch
        # distances = torch.linalg.norm(shift_points, dim=1, keepdim=True)
        spc = kal.ops.conversions.unbatched_pointcloud_to_spc(shift_points, space_carving_level)

        shift_points_directions = (shift_points/(shift_points**2).sum(1).sqrt().reshape(-1,1))
        virtual_origin = -shift_points_directions*3
            
        octree, point_hierarchy, pyramid, prefix = spc.octrees, spc.point_hierarchies, spc.pyramids[0], spc.exsum
        nugs_ridx, nugs_pidx, depth = kal.render.spc.unbatched_raytrace(octree, point_hierarchy, pyramid, prefix, \
                                                                            virtual_origin, shift_points_directions, space_carving_level, with_exit=True)

        stop_depth =  (shift_points**2).sum(1).sqrt() - stop_depth_thre + 3.0
        mask = (depth[:,0]>3.0) & (depth[:,1]<stop_depth[nugs_ridx.long()]) & ((depth[:,1] - depth[:,0])> inter_dist_thre)
   
        steps = torch.rand(mask.sum().item(),1).cuda() # randomly sample one point on each intersected segment 
        origins = virtual_origin[nugs_ridx[mask].long()]
        directions = shift_points_directions[nugs_ridx[mask].long()]
        depth_range = depth[mask,1] - depth[mask,0]

        space_carving_samples = origins + directions*((depth[mask,0] + steps.reshape(1,-1)*depth_range).reshape(-1,1))

        space_carving_labels = torch.zeros(space_carving_samples.shape[0], device=self.dev) # all as 0 (free)

        return space_carving_samples, space_carving_labels
