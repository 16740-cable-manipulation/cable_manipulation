from rs_driver import Realsense
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    realsense = Realsense()
    depth, rgb = realsense.getFrameSet()
    realsense.close()
    new = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    output = Image.fromarray(new, "RGB")
    output.save("cable_manipulation/cableImages/testImage.jpg")
    plt.imshow(new)
    plt.show()
    # dp = depth.reshape((-1, 3))
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(dp)
    # o3d.visualization.draw_geometries([pc])
