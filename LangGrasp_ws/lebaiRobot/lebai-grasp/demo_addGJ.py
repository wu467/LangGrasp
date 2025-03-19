import time
import lebai_sdk
import math
import numpy as np
import transforms3d as tfs
from GPT import multiple_gpt as gpt_multiple_infer
import nest_asyncio
nest_asyncio.apply()


# 初始化机械臂
def init_robot():
    lebai_sdk.init()
    robot_ip = "X.X.X.X"
    lebai = lebai_sdk.connect(robot_ip, False)
    return lebai


# Generate a homogeneous matrix
def homogeneous_generate(location, degrees=0):
    x, y, z, rx, ry, rz = location['x'], location['y'], location['z'], location['rx'], location['ry'], location['rz']
    if degrees == 1:
        rx = math.radians(rx)
        ry = math.radians(ry)
        rz = math.radians(rz)
    rotation = tfs.euler.euler2mat(rx, ry, rz, "sxyz")   # R = Rz * Ry * Rx
    translation = np.asarray([x, y, z])
    homogeneous_matrix = tfs.affines.compose(translation, rotation, [1, 1, 1])
    return homogeneous_matrix


# Homogeneous matrix operations
def get_target_position(T_e2b, T_c2e, T_t2c):
    bh = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    T = T_c2e @ T_t2c
    r1 = T[:3, :3]
    t1 = T[:3, 3]
    r2 = r1 @ bh
    T2 = np.eye(4)
    T2[:3, :3] = r2
    T2[:3, 3] = t1
    result = T_e2b @ T2
    R = result[:3, :3]
    t = result[:3, 3]
    print("Object relative to the end tool of the manipulator:", T2)
    rx, ry, rz = tfs.euler.mat2euler(R, 'sxyz')
    return rx, ry, rz, t[0], t[1], t[2]


def get_target2camera(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    print("GsNet Matrix: ",  transformation_matrix)
    return transformation_matrix


# The manipulator moves to the target point and grabs
def grasp(lebai, motion_id, rx, ry, rz, x, y, z):
    state = lebai.get_motion_state(motion_id)
    while state != "FINISHED":
        time.sleep(0.1)
        state = lebai.get_motion_state(motion_id)
    cartesian_pose = {'x': x, 'y': y, 'z': z, 'rz': rz, 'ry': ry, 'rx': rx}
    a = 0.3
    v = 0.1
    t = 0
    r = 0
    lebai.movel(cartesian_pose, a, v, t, r)
    lebai.wait_move()


# Move to the camera position
def move_to_camera(lebai):
    cartesian_pose = {'x': -0.2815, 'y': -0.00869, 'z':  0.37819, 'rz': 3.0584, 'ry': -0.04039, 'rx': 3.09896}
    a = 0.2
    v = 0.1
    t = 0
    r = 0
    lebai.movel(cartesian_pose, a, v, t, r)
    lebai.wait_move()
    # Get robot arm motion data (actual tcp pose: tool relative to base)
    kin_data = lebai.get_kin_data()
    actual_tcp_pose = kin_data['actual_tcp_pose']
    print('actual_flange_pose:', kin_data['actual_flange_pose'])
    print('actual_tcp_pose：', actual_tcp_pose)

    motion_id = lebai.get_running_motion()
    return actual_tcp_pose, motion_id


if __name__ == '__main__':
    grasp_label, translation, rotation, score = gpt_multiple_infer.gpt_infer()
    print("grasp_label: ", grasp_label)
    print("translation: ", translation)
    print("rotation: ", rotation)
    print("score: ", score)