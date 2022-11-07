#!/home/student/Prog/iamEnv/bin/python
"""cable_simple_demo

1. Home the arm
2. Take an image and determine where to grasp
3. Command arm to grasp
4. Command arm to move then ungrasp
"""
from math import atan2
import time
import numpy as np
from rs_driver import Realsense
from autolab_core import RigidTransform
from frankapy import FrankaArm
import yaml
from scipy.spatial.transform import Rotation as R
import copy

SMALL_Z_OFFSET = 0.002

if __name__ == "__main__":
    T_et_ee = np.eye(4)  # transformation from fingertip of ee to center of ee
    T_et_ee[3, 3] = -0.1149  # only z translation
    # Step 0: load T_ee_c
    T_ee_c = np.eye(4)
    with open("cable_manipulation/test.yaml") as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
        R_ee_c = R.from_quat(np.array(loaded["rot_e_c"]))
        T_ee_c[:3, :3] = R_ee_c.as_matrix()
        T_ee_c[:3, 3] = np.array(loaded["trans_e_c"])
        print(T_ee_c)

    realsense = Realsense()
    fa = FrankaArm()
    # Step 1: home the arm
    fa.reset_joints()
    time.sleep(4)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    trans = np.array([0.3, 0.0, 0.4])
    home_pose = RigidTransform(
        rotation=rot,
        translation=trans,
        from_frame="franka_tool",
        to_frame="world",
    )
    fa.goto_pose(home_pose, use_impedance=False)
    time.sleep(4)

    # Step 2:
    # 2.1 Take image
    vals = realsense.getFrameSet()
    if vals is None:
        RuntimeError("Failed to get frameset")
        quit(1)
    depth, bgr = vals

    # 2.2 Process image to get 2D coordinate and tangent
    color = "blue"
    pixel, vec_c = process_image(bgr, color)
    vec_c_3d = np.array([[vec_c[0], vec_c[1], 0]])

    # 2.3 Get 3D coordinate
    point_c = realsense.deproject_pixel(depth, pixel[0], pixel[1])
    if (
        point_c[2] < 0.05 or point_c[2] > 1
    ):  # filter out points with wrong depth
        RuntimeError("Bad pixel, no valid 3D coordinate!")
    # 2.4 Compute gripper pose
    # transform point to world frame
    tf_w_ee = fa.get_pose()
    t_w_ee = tf_w_ee.translation
    q_w_ee = tf_w_ee.quaternion  # xyzw
    R_w_ee = R.from_quat(q_w_ee).as_matrix()
    T_w_ee = np.eye(4)
    T_w_ee[:3, :3] = R_w_ee
    T_w_ee[:3, 3] = t_w_ee
    point_w = T_w_ee * T_ee_c * point_c
    vec_w = R_w_ee * R_ee_c * vec_c_3d
    # translation has to do with tip of ee
    # orientation is determined based on tangent
    T_w_et = np.eye(4)
    t_w_et = copy.deepcopy(point_w)
    t_w_et[2] += SMALL_Z_OFFSET
    T_w_et[:3, 3] = t_w_et
    r_w_ee0 = R.from_euler("zyx", [0, 0, np.pi])
    angle = -atan2(vec_w[1], vec_w[0])
    if angle > np.pi / 2:
        angle -= np.pi
    if angle < -np.pi / 2:
        angle += np.pi
    r_ee0_et = R.from_euler("zyx", [angle, 0, 0])
    R_w_et = np.matmul(r_w_ee0.as_matrix(), r_ee0_et.as_matrix())
    T_w_et[:3, :3] = R_w_et
    T_w_ee_out = np.matmul(T_w_et, T_et_ee)

    # Step 3: Command arm to reach target and grasp
    grasp_pose = RigidTransform(
        rotation=T_w_ee_out[:3, :3],
        translation=T_w_ee_out[:3, 3],
        from_frame="franka_tool",
        to_frame="world",
    )
    fa.goto_gripper(0.02)
    time.sleep(2)
    fa.goto_pose(grasp_pose, use_impedance=False)
    time.sleep(4)

    # Step 4: Lift, move, lower, ungrasp, home
