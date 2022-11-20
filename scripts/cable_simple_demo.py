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
from cable_manipulation import CableManipulation
from rs_driver import Realsense
from autolab_core import RigidTransform
from frankapy import FrankaArm
from scipy.spatial.transform import Rotation as R
import copy
from my_franka import MyFranka
import cv2


if __name__ == "__main__":
    fa = MyFranka()
    cable_manipulator = CableManipulation(640, 480, use_rs=True)
    fa.reset_joint_and_gripper()

    fa.goto_capture_pose()

    # Step 2:
    # 2.1 Take image
    vals = cable_manipulator.realsense.getFrameSet(skip_frames=5)
    if vals is None:
        RuntimeError("Failed to get frameset")
        quit(1)
    depth, bgr = vals
    print("depth_shape: ", depth.shape)
    print("bgr_shape: ", bgr.shape)

    # 2.2 Process image to get 2D coordinate and tangent
    color = "blue"
    mask, pixel, vec = cable_manipulator.processImage(
        bgr, color, inputDepth=depth, visualize=True
    )

    # 2.3 Get 3D coordinate
    point_c = cable_manipulator.realsense.deproject_pixel(
        depth, pixel[0], pixel[1]
    )
    if (
        point_c[2] < 0.05 or point_c[2] > 1
    ):  # filter out points with wrong depth
        RuntimeError("Bad pixel, no valid 3D coordinate!")

    ptprime_tmp = [pixel[0] + vec[0], pixel[1] + vec[1]]
    print(ptprime_tmp)
    ptprime_tmp = cable_manipulator.realsense.deproject_pixel(
        depth, ptprime_tmp[0], ptprime_tmp[1]
    )
    if (
        ptprime_tmp[2] < 0.05 or ptprime_tmp[2] > 1
    ):  # filter out points with wrong depth
        RuntimeError("Bad vector, no valid 3D coordinate!")
    vec_c_3d = np.array(
        [[ptprime_tmp[0] - point_c[0], ptprime_tmp[1] - point_c[1], 0]]
    )
    print("point_c: ", point_c)
    print("vec_c_3d: ", vec_c_3d)

    # 2.4 Compute gripper pose
    # Step 3: Command arm to reach target and grasp
    fa.goto_point_and_vec(point_c, vec_c_3d)

    # Step 4: Lift, move, lower, ungrasp, home
    cur_p = fa.get_pose()
    cur_R = cur_p["R"]
    cur_t = cur_p["t"]
    lift_t = cur_t + [0, 0, 0.2]
    move_t = lift_t + [0.2, 0, 0]
    lower_t = move_t + [0, 0, -0.2]
    fa.goto_pose({"R": cur_R, "t": lift_t})
    time.sleep(fa.time_per_move)
    fa.goto_pose({"R": cur_R, "t": move_t})
    time.sleep(fa.time_per_move)
    fa.goto_pose({"R": cur_R, "t": lower_t})
    time.sleep(fa.time_per_move)
    fa.reset_joint_and_gripper()
