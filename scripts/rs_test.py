from rs_driver import Realsense
import cv2
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    realsense = Realsense()
    depth, rgb = realsense.getFrameSet()
    realsense.close()
    dp = depth.reshape((-1, 3))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(dp)
    o3d.visualization.draw_geometries(
        [pc],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
    )
