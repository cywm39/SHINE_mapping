import yaml
import os
import torch
import numpy as np
from typing import List

class Gaussian_config:
    def __init__(self):
        self.pose_file_path: str = ''
        self.image_folder_path: str = ''
        self.pc_path: str = ''

        self.spatial_lr_scale: float = 1.0
        self.position_lr_max_steps: int = 1000
        self.device: str = 'cuda'
        self.name: str = "dummy"  # experiment name
        self.output_root: str = ""  # output root folder
        self.xyz_lr: float = 0.00016
        self.f_dc_lr: float = 0.0025
        self.f_rest_lr: float = 0.0025/20.0
        self.opacity_lr: float = 0.05
        self.scaling_lr: float = 0.005
        self.rotation_lr: float = 0.001
        self.densification_interval: int = 0
        self.opacity_reset_interval: int = 0
        self.densify_from_iter: int = 0
        self.densify_until_iter: int = 0
        self.densify_grad_threshold: float = 0.0002
        self.lambda_dssim: float = 0.2
        self.white_background: bool = False
        self.out_result_iter: int = 100

    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        self.pose_file_path = config_args['pose_file_path']
        self.image_folder_path = config_args['image_folder_path']
        self.pc_path = config_args['pc_path']
        self.spatial_lr_scale = config_args['spatial_lr_scale']
        self.position_lr_max_steps = config_args['position_lr_max_steps']
        self.device = config_args['device']
        self.name = config_args['name']
        self.output_root = config_args['output_root']
        self.xyz_lr = config_args['xyz_lr']
        self.f_dc_lr = config_args['f_dc_lr']
        self.f_rest_lr = config_args['f_rest_lr']
        self.opacity_lr = config_args['opacity_lr']
        self.scaling_lr = config_args['scaling_lr']
        self.rotation_lr = config_args['rotation_lr']

        self.densification_interval = self.position_lr_max_steps//300
        self.opacity_reset_interval = self.position_lr_max_steps//10
        self.densify_from_iter = self.position_lr_max_steps//6
        self.densify_until_iter = self.position_lr_max_steps//2
        self.densify_grad_threshold = config_args['densify_grad_threshold']

        self.lambda_dssim = config_args['lambda_dssim']
        self.white_background = config_args['white_background']
        self.out_result_iter = config_args['out_result_iter']
