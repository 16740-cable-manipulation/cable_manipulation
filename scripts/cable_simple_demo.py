#!/home/student/Prog/iamEnv/bin/python
"""cable_simple_demo

1. Home the arm
2. Take an image and determine where to grasp
3. Command arm to grasp
4. Command arm to move then ungrasp
"""
import time
import numpy as np
from rs_driver import Realsense
from autolab_core import RigidTransform
from frankapy import FrankaArm
import yaml
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    T_et_ee = np.eye(4)  # transformation from fingertip of ee to center of ee
    T_et_ee[3, 3] = -0.1  # only z translation
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
    point_c = depth[pixel[0], pixel[1], :]

    # 2.4 Compute gripper pose
    # transform point to world frame

    point_w = T_w_ee * T_ee_c * point_c
    vec_w = R_w_ee * R_ee_c * vec_c_3d
    # orientation is determined based on tangent

    # Step 3: Command arm to reach target and grasp
    grasp_pose = RigidTransform(
        rotation=rot,
        translation=trans,
        from_frame="franka_tool",
        to_frame="world",
    )
    fa.goto_pose(grasp_pose, use_impedance=False)
    time.sleep(4)

    # Step 4: Lift, move, lower, ungrasp, home
