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
    pc2, rgb = vals
    # 2.2 Process image to get 2D coordinate and tangent
    pixel, tangent = process_image(rgb)
    # 2.3 Get 3D coordinate
    point = realsense.get3DCoord(pixel)
    # 2.4 Compute gripper pose
    # transform point to world frame

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
