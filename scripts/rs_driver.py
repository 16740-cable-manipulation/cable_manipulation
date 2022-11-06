import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

import pyrealsense2 as rs


class Realsense:
    def __init__(self):
        self.seq = 0
        self.time_offset = None
        self.fnumber_captured = False
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth)
        self.cfg.enable_stream(rs.stream.color)

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.pc = rs.pointcloud()
        self.frame_buffer = []
        self.bridge = CvBridge()

        self.setSyncMode()
        self.start()

    def start(self):
        self.pipeline.start(self.cfg)
        print("Realsense ready!")

    def setSyncMode(self):
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        r = self.cfg.resolve(pipeline_wrapper)
        device = r.get_device()
        sensor = device.query_sensors()[0]
        # set sensor sync mode
        sensor.set_option(rs.option.inter_cam_sync_mode, 0)

    def getFrameSet(self):
        """Get rs frames, construct PointCloud2 msg, add to buffer"""
        frameset = self.pipeline.wait_for_frames()
        vals = self.getAlignedFrames(frameset)
        return vals

    def getAlignedFrames(self, frameset):
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frameset)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        self.pc.map_to(color_frame)
        points = self.pc.calculate(aligned_depth_frame)
        pc2 = self.to_pointcloud2(points, color_image, self.seq)
        return pc2, color_image

    def get3DCoord(self, px, py):
        """Get the 3D coords of a pixel in color image"""

        pass

    def to_pointcloud2(self, rs_points, rs_texture):
        h, w = rs_texture.shape[0:2]

        v, t = rs_points.get_vertices(), rs_points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        # check depth validity
        valid_verts_idx = np.argwhere(verts[:, 2] > 0.0).flatten()
        points_xyz = verts[valid_verts_idx]

        texcoords = texcoords[valid_verts_idx]
        valid_tex_mask = (
            (texcoords[:, 0] >= 0.0)
            & (texcoords[:, 0] < 1.0)
            & (texcoords[:, 1] >= 0.0)
            & (texcoords[:, 1] < 1.0)
        ).flatten()
        valid_tex_idx = np.argwhere(valid_tex_mask).flatten()

        texcoords_valid = texcoords[valid_tex_idx]
        texcoords_valid[:, 0] = texcoords_valid[:, 0] * w
        texcoords_valid[:, 1] = texcoords_valid[:, 1] * h
        texcoords_valid = np.floor(texcoords_valid).astype(np.uint32)
        bgr = np.zeros((len(valid_tex_mask), 1), dtype=np.uint32)
        tmp = np.full((len(valid_tex_idx), 1), 255, dtype=np.uint8)
        tex = rs_texture[texcoords_valid[:, 1], texcoords_valid[:, 0]].astype(
            np.uint8
        )
        # turn tex (rgb) into bgr
        bgr_at_valid_tex_idx = np.hstack((np.flip(tex, axis=1), tmp))
        bgr_at_valid_tex_idx = np.ascontiguousarray(
            bgr_at_valid_tex_idx, dtype=np.uint8
        )

        bgr[valid_tex_idx] = bgr_at_valid_tex_idx.view(np.uint32)

        data = np.zeros(
            len(points_xyz),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ],
        )
        data["x"] = points_xyz[:, 0]
        data["y"] = points_xyz[:, 1]
        data["z"] = points_xyz[:, 2]
        data["rgb"] = bgr.flatten()
        return msg

    def close(self):
        print("Closing Realsense")
        self.pipeline.stop()
