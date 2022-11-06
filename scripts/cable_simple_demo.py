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

if __name__ == "__main__":
    # Step 0: load T_ee_c
    T_ee_c = 0
    realsense = Realsense()
    fa = FrankaArm()
    # Step 1: home the arm
    fa.reset_joints()
    time.sleep(4)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    trans = np.array([0.3, -0.2, 0.4])
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
    depth, rgb = vals

    # 2.2 Process image to get 2D coordinate and tangent
    pixel, vec_c = process_image(rgb)
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
