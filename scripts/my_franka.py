from math import atan2
import time
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import yaml
from scipy.spatial.transform import Rotation as R
import copy
from action import Action


class MyFranka:
    def __init__(self):
        self.time_per_move = 3
        self.small_z_offset = -0.015
        self.fa = FrankaArm()
        # transformation from fingertip of ee to center of ee
        self.T_ee_et = np.eye(4)
        self.tz_ee_et = 0.1149  # only z translation
        self.T_ee_et[3, 3] = self.tz_ee_et
        self.T_ee_c = self.load_extrinsic()

    def load_extrinsic(self):
        # Step 0: load T_ee_c
        T_ee_c = np.eye(4)
        T_ee_c[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T_ee_c[:3, 3] = np.array([0.0412, -0.0326, 0.06775 - self.tz_ee_et])
        print("T_ee_c from calibration:")
        print(T_ee_c)
        return T_ee_c

    def reset_joint_and_gripper(self):
        self.fa.open_gripper()
        time.sleep(2)
        self.fa.reset_joints()
        time.sleep(self.time_per_move)

    def goto_pose(self, pose, use_impedance=False):
        assert type(pose) is dict
        rot = pose["R"]
        trans = pose["t"]
        if rot.shape != (3, 3):
            rot = R.from_quat(rot).as_matrix()
        p = RigidTransform(
            rotation=rot,
            translation=trans,
            from_frame="franka_tool",
            to_frame="world",
        )
        self.fa.goto_pose(p, use_impedance=use_impedance)

    def goto_capture_pose(self):
        home_pose = {
            "R": np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            ),
            "t": np.array([0.3, 0.0, 0.4]),
        }
        self.goto_pose(home_pose, use_impedance=False)
        time.sleep(self.time_per_move)

    def get_pose(self):
        """Pose of tip of ee in world frame"""
        tf_w_ee = self.fa.get_pose()
        t_w_ee = tf_w_ee.translation
        q_w_ee = tf_w_ee.quaternion  # xyzw
        q_w_ee = np.array([q_w_ee[1], q_w_ee[2], q_w_ee[3], q_w_ee[0]])
        R_w_ee = R.from_quat(q_w_ee).as_matrix()
        p = {"R": R_w_ee, "t": t_w_ee}
        print(p)
        return p

    def goto_point_and_vec(self, point_c, vec_c_3d):
        # transform point to world frame
        tf_w_ee = self.get_pose()  # actually is tf_w_et
        T_w_ee = np.eye(4)
        T_w_ee[:3, :3] = tf_w_ee["R"]
        T_w_ee[:3, 3] = tf_w_ee["t"]
        print("translation: ", tf_w_ee["t"])
        point_c_homo = np.reshape(np.hstack((point_c, np.ones(1))), (-1, 1))
        print(point_c_homo)
        p_et = np.matmul(self.T_ee_c, point_c_homo)
        print(p_et)
        point_w = np.matmul(T_w_ee, p_et).flatten()[:3]
        print(point_w)
        vec_w = (
            np.matmul(
                np.matmul(tf_w_ee["R"], self.T_ee_c[:3, :3]),
                np.reshape(vec_c_3d, (-1, 1)),
            )
        ).flatten()
        # translation has to do with tip of ee
        # orientation is determined based on tangent
        T_w_et = np.eye(4)
        t_w_et = copy.deepcopy(point_w)
        t_w_et[2] += self.small_z_offset
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
        p = {"R": T_w_et[:3, :3], "t": T_w_et[:3, 3]}
        print("T_w_et: ")
        print(T_w_et)
        self.fa.goto_gripper(0.05)
        time.sleep(2)
        self.goto_pose(p, use_impedance=False)
        time.sleep(self.time_per_move)
        self.fa.close_gripper()
        time.sleep(2)

    def exe_action(action: Action):
        # grasp action.pick_3d

        # lift to action.z

        # move to a point above action.place_3d, with height z

        # lower to table

        # ungrasp
        pass
