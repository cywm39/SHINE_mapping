import yaml 
import numpy as np
file = '/home/cy/NeRF/shine_mapping_input/r3live_config.yaml'

with open(file, 'r') as input:
    data = yaml.safe_load(input)
    print(data)
    matrix = data['r3live_vio']['camera_ext_matrix']
    print(matrix)