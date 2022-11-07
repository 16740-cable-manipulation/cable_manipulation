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
        self.K = None
        self.D = None
        self.intr = None  # depth intrinsic
        self.depth_scale = 1.0

        self.setSyncMode()
        self.start()

    def start(self):
        profile = self.pipeline.start(self.cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.intr = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        print(self.intr)
        self.K = np.array(
            [
                [self.intr.fx, 0.0, self.intr.ppx],
                [0.0, self.intr.fy, self.intr.ppy],
                [0.0, 0.0, 1],
            ]
        )
        self.D = np.array(self.intr.coeffs)
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
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        h, w = color_image.shape[0:2]
        depth_image = (
            np.asanyarray(aligned_depth_frame.get_data()) * self.depth_scale
        )
        depth_image = np.where(depth_image > 0.0, depth_image, 0.0)
        return depth_image, color_image

    def deproject_pixel(self, depth_image, px, py):
        depth = depth_image[py, px]
        pt = np.zeros(3)
        if depth > 0.0:
            point = rs.rs2_deproject_pixel_to_point(self.intr, [px, py], depth)
            pt = np.array([point[0], point[1], point[2]], dtype=np.float)
        return pt

    def to_depth_map(self, rs_points, w, h):
        v, t = rs_points.get_vertices(), rs_points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        depth_map = np.zeros((h, w, 3), dtype=np.float)

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
        points_xyz_valid = points_xyz[valid_tex_idx]
        # for every (uv) coord (idx = i) in texcoord_valid, get the
        # point_xyz_valid and assign to depth map's (uv) pixel
        depth_map[
            texcoords_valid[:, 1], texcoords_valid[:, 0], :
        ] = points_xyz_valid

        return depth_map

    def close(self):
        print("Closing Realsense")
        self.pipeline.stop()
