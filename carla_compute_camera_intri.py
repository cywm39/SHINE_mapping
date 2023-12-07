import numpy as np


# 计算原始相机原始内参
image_w = 800 # b c h[2] w[3] # 原始内参
image_h = 600 # b c h[2] w[3] # 原始内参
fov = 90 # 原始内参
focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0)) # Focus_length
# In this case Fx and Fy are the same since the pixel aspect
# ratio is 1
origin_intrinsic_rgball_K = np.identity(3)
origin_intrinsic_rgball_K[0, 0] = origin_intrinsic_rgball_K[1, 1] = focal
origin_intrinsic_rgball_K[0, 2] = image_w / 2.0
origin_intrinsic_rgball_K[1, 2] = image_h / 2.0

print(origin_intrinsic_rgball_K)