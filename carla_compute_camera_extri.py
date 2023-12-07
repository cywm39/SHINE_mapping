import numpy as np
from scipy.spatial.transform import Rotation as R

# def create_transformation_matrix(translation, rotation):
#     # 创建变换矩阵
#     matrix = np.eye(4)
#     matrix[:3, 3] = translation
#     matrix[:3, :3] = R.from_euler('xyz', rotation, degrees=True).as_matrix()
#     return matrix

# # 相机外部参数
# camera_translation = np.array([3.5, 0.2, 2.55]) # 这三个的顺序就是xyz
# camera_rotation = np.array([20.0, 0.0, 0.0]) # 这三个的顺序是pitch在最前面, 后面两个可能是[roll, yaw], 也可能是反过来的

# # 雷达外部参数
# lidar_translation = np.array([3.5, -0.2, 2.55])
# lidar_rotation = np.array([0.0, 0.0, 0.0])

# # 创建相机和雷达的变换矩阵
# camera_matrix = create_transformation_matrix(camera_translation, camera_rotation)
# lidar_matrix = create_transformation_matrix(lidar_translation, lidar_rotation)

# # 计算相机到雷达的相对变换矩阵
# camera_to_lidar_matrix = np.linalg.inv(camera_matrix) @ lidar_matrix

# print("Camera to Lidar Transformation Matrix:")
# print(camera_to_lidar_matrix)


def E_Calc(roll, pitch, yaw, x, y, z):
    yaw = np.radians(yaw)
    roll = np.radians(roll)
    pitch = np.radians(pitch)

    yaw_matrix = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitch_matrix = np.matrix([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])

    roll_matrix = np.matrix([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    rotation_matrix = yaw_matrix * pitch_matrix * roll_matrix

    # translation_matrix = np.matrix([
    # [x],
    # [y],
    # [z]
    # ])
    translation_matrix = np.array([x, y, z])
    # 拼接旋转和平移矩阵并增加0,0,0,1
    extrinsic_matrix = np.hstack([np.array(rotation_matrix), np.array([translation_matrix]).T])
    extrinsic_matrix = np.vstack([np.array(extrinsic_matrix), np.array([[0,0,0,1]]) ])
    return extrinsic_matrix

camera = E_Calc(np.radians(20.0), 0.0, 0.0, 3.5, 0.2, 2.55)
lidar = E_Calc(0.0, 0.0, 0.0, 3.5, -0.2, 2.55)
print(np.linalg.inv(camera) @ lidar)

# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # 相机相对车身的坐标和旋转角度
# cam_spawn_point = {"x": 3.5, "y": 0.2, "z": 2.55, "roll": 0.0, "pitch": 20.0, "yaw": 0.0}

# # 雷达相对车身的坐标和旋转角度
# lidar_spawn_point = {"x": 3.5, "y": -0.2, "z": 2.55, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

# # 将角度转换为弧度
# roll, pitch, yaw = np.radians(cam_spawn_point["roll"]), np.radians(cam_spawn_point["pitch"]), np.radians(cam_spawn_point["yaw"])
# # 构建相机的变换矩阵
# cam_rotation = R.from_euler('xyz', [pitch, roll, yaw]).as_matrix()
# cam_translation = np.array([cam_spawn_point["x"], cam_spawn_point["y"], cam_spawn_point["z"]])

# # 类似地，构建雷达的变换矩阵
# lidar_roll, lidar_pitch, lidar_yaw = np.radians(lidar_spawn_point["roll"]), np.radians(lidar_spawn_point["pitch"]), np.radians(lidar_spawn_point["yaw"])
# lidar_rotation = R.from_euler('xyz', [lidar_pitch, lidar_roll , lidar_yaw]).as_matrix()
# lidar_translation = np.array([lidar_spawn_point["x"], lidar_spawn_point["y"], lidar_spawn_point["z"]])

# # 计算相机到雷达的变换矩阵
# cam_to_lidar = np.eye(4)  # 初始化为单位矩阵
# cam_to_lidar[:3, :3] = np.dot(lidar_rotation.T, cam_rotation)  # 旋转：雷达坐标系的逆与相机坐标系的转换
# cam_to_lidar[:3, 3] = cam_translation - np.dot(cam_to_lidar[:3, :3], lidar_translation)  # 平移

# print(cam_to_lidar)
