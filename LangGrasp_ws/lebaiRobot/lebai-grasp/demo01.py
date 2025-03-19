import time

import lebai_sdk
import math
import numpy as np
import transforms3d as tfs


# If you get an event Loop-related RuntimeError, try adding the following two lines
import nest_asyncio
nest_asyncio.apply()


# initialize the manipulator
def init_robot():
    lebai_sdk.init()
    # set the robot ip address
    robot_ip = "10.20.17.1"
    # creating a robot instance
    lebai = lebai_sdk.connect(robot_ip, False)
    return lebai

# generate a homogeneous matrix
def homogeneous_generate(location, degrees=0):
    x, y, z, rx, ry, rz = location['x'], location['y'], location['z'], location['rx'], location['ry'], location['rz']
    # If the Euler Angle is in degrees, it is converted to radians
    if degrees == 1:
        rx = math.radians(rx)
        ry = math.radians(ry)
        rz = math.radians(rz)
    rotation = tfs.euler.euler2mat(rx, ry, rz, "sxyz")   # R = Rz * Ry * Rx
    translation = np.asarray([x, y, z])
    # The resultant homogeneous matrix
    homogeneous_matrix = tfs.affines.compose(translation, rotation, [1, 1, 1])
    return homogeneous_matrix


# Homogeneous matrix operations
def get_target_position(T_e2b, T_c2e, T_t2c):
    result = T_e2b @ T_c2e @ T_t2c
    # The rotation matrix and the offset vector are extracted
    R = result[:3, :3]
    t = result[:3, 3]
    # The rotation matrix turns the Euler Angle
    rx, ry, rz = tfs.euler.mat2euler(R, 'sxyz')
    return rx, ry, rz, t[0], t[1], t[2]


# The rotation matrix and offset vector obtained by GsNet are synthesized into a homogeneous matrix
def get_target2camera(rotation_matrix, translation_vector):
    # Rotate 90 degrees around the Y-axis
    bh = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]])
    rotation_matrix = rotation_matrix @ bh
    # Create a 4 x 4 homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    # Put the rotation matrix into the top-left 3 x 3 part of the homogeneous transformation matrix
    transformation_matrix[:3, :3] = rotation_matrix
    # Put the offset vector into the last column of the first three rows of the homogeneous transformation matrix
    transformation_matrix[:3, 3] = translation_vector
    print("GsNet Matrix: ",  transformation_matrix)
    return transformation_matrix


# The manipulator moves to the target point and grabs
def grasp(lebai, motion_id, rx, ry, rz, x, y, z):
    state = lebai.get_motion_state(motion_id)
    while state != "FINISHED":
        time.sleep(0.1)  # check the status every 100ms
        state = lebai.get_motion_state(motion_id)

    # cartesian data of the target pose
    cartesian_pose = {'x': x, 'y': y, 'z': z, 'rz': rz, 'ry': ry, 'rx': rx}
    a = 0.3  # spatial acceleration m s2
    v = 0.1  # Space velocity (m/s)
    t = 0    # Movement time (s). When t > 0, the parameters velocity v and acceleration a are invalid
    r = 0    # Blending radius (m). Smoothing effect used to specify paths
    lebai.movel(cartesian_pose, a, v, t, r)  # 直线运动
    lebai.wait_move()


# Move to the camera position
def move_to_camera(lebai):
    # Photo pose Cartesian data
    cartesian_pose = {'x': -0.233, 'y': -0.108, 'z': 0.641, 'rz': 3.117, 'ry': -0.169, 'rx': 3.117}
    a = 0.3
    v = 0.1
    t = 0
    r = 0
    lebai.movel(cartesian_pose, a, v, t, r)
    lebai.wait_move()  # Wait for the movement to complete
    # Acquisition of robot arm motion data (actual_tcp_pose: end tool relative to base, actual_flange_pose: end relative to base）
    kin_data = lebai.get_kin_data()
    actual_tcp_pose = kin_data['actual_tcp_pose']
    print('actual_flange_pose:', kin_data['actual_flange_pose'])
    print('actual_tcp_pose：', actual_tcp_pose)
    motion_id = lebai.get_running_motion()
    return actual_tcp_pose, motion_id


if __name__ == '__main__':
    lebai = init_robot()
    lebai.start_sys()
    lebai.init_claw()
    lebai.set_tcp({'x': 0, 'y': 0, 'z': 0, 'rz': 0, 'ry': 0, 'rx': 0})
    # 1.The robot arm moves to the photo position, and the pose of the end relative to the base is obtained (the euler Angle unit is radians by calling the SDK) and the motion_id
    end2base, motion_id = move_to_camera(lebai)
    # 2.Camera pose (in angles) with respect to the end tool coordinate system.
    camera2end = {'x': -0.10145, 'y': 0.00155198, 'z': -0.175, 'rx': -6.02255, 'ry': -0.406039, 'rz': -87.0701}
    # 3.The pose of the object relative to the camera (obtained by calling Gs Net)
    # get_picture()  # Take pictures (rgb and depth)
    translation = np.array([0.09437691, -0.12824655, 0.61])
    rotation = np.array([[0.12232582, -0.983852,  -0.13065802],
                         [0.6724109,  0.17898357, -0.718212],
                         [0.73,       0,         0.6834472]])
    # 4.The three poses are transformed into orthogonal matrices
    T_e2b = homogeneous_generate(end2base)
    T_c2e = homogeneous_generate(camera2end, degrees=1)
    T_t2c = get_target2camera(rotation, translation)
    # 5.Matrix operation, get T t 2 b and convert to (x,y,z,rx,ry,rz)
    rx, ry, rz, x, y, z = get_target_position(T_e2b, T_c2e, T_t2c)
    print("rx:", math.degrees(rx))
    print("ry: ", math.degrees(ry))
    print("rz: ", math.degrees(rz))
    print("x: ", x)
    print("y: ", y)
    print("z: ", z)
    # 6.Move to the target position and grasp
    grasp(lebai, motion_id, rx, ry, rz, x, y, z+0.2)
    # 7.Stop the arm
    lebai.stop_sys()
