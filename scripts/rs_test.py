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
    o3d.visualization.draw_geometries([pc])
