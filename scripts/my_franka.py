from math import atan2
import time
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import yaml
from scipy.spatial.transform import Rotation as R
import copy


class MyFranka:
    def __init__(self):
        self.time_per_move = 3
        self.small_z_offset = 0.002
        self.fa = FrankaArm()
        # transformation from fingertip of ee to center of ee
        self.T_et_ee = np.eye(4)
        self.tz_ee_et = -0.1149  # only z translation
        self.T_ee_c = self.load_extrinsic()

    def load_extrinsic(self):
        # Step 0: load T_ee_c
        T_ee_c = np.eye(4)
        T_ee_c[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T_ee_c[:3, 3] = np.array([0.0412, -0.0326, 0.06775 + self.tz_ee_et])
        print("T_ee_c from calibration:")
        print(T_ee_c)
        return T_ee_c

    def reset_joint_and_gripper(self):
        self.fa.reset_joints()
        self.fa.open_gripper()
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

    def goto_point_and_vec(self, point_c, vec_c_3d):
        # transform point to world frame
        tf_w_ee = self.fa.get_pose()
        t_w_ee = tf_w_ee.translation
        print("translation: ", t_w_ee)
        q_w_ee = tf_w_ee.quaternion  # xyzw
        R_w_ee = R.from_quat(q_w_ee).as_matrix()
        T_w_ee = np.eye(4)
        T_w_ee[:3, :3] = R_w_ee
        T_w_ee[:3, 3] = t_w_ee
        point_c_homo = np.reshape(np.hstack((point_c, np.ones(1))), (-1, 1))
        print(point_c_homo)
        point_w = np.matmul(
            np.matmul(T_w_ee, self.T_ee_c), point_c_homo
        ).flatten()[:3]
        print(point_w)
        vec_w = (
            np.matmul(
                np.matmul(R_w_ee, self.T_ee_c[:3, :3]),
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
        self.fa.goto_gripper(0.02)
        time.sleep(2)
        self.goto_pose(p, use_impedance=False)
        time.sleep(self.time_per_move)
