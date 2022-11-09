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
        self.T_et_ee[3, 3] = -0.1149  # only z translation
        self.T_ee_c = self.load_extrinsic()

    def load_extrinsic(self):
        # Step 0: load T_ee_c
        T_ee_c = np.eye(4)
        with open("cable_manipulation/test.yaml") as f:
            loaded = yaml.load(f, Loader=yaml.Loader)
            R_ee_c = R.from_quat(np.array(loaded["rot_e_c"]))
            T_ee_c[:3, :3] = R_ee_c.as_matrix()
            T_ee_c[:3, 3] = np.array(loaded["trans_e_c"])
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
        self.fa.goto_pose(p, use_impedance)

    def goto_capture_pose(self):
        home_pose = {
            "R": np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            ),
            "t": np.array([0.3, 0.0, 0.4]),
        }
        self.goto_pose(home_pose, use_impedance=False)
        time.sleep(self.time_per_move)

    def my_get_pose(self):
        tf_w_ee = self.fa.get_pose()
        t_w_ee = tf_w_ee.translation
        q_w_ee = tf_w_ee.quaternion  # xyzw
        R_w_ee = R.from_quat(q_w_ee).as_matrix()
        p = {"R": R_w_ee, "t": t_w_ee}
        return p



    def goto_point_and_vec(self, point_c, vec_c_3d):
        # transform point to world frame
        tf_w_ee = self.fa.get_pose()
        t_w_ee = tf_w_ee.translation
        q_w_ee = tf_w_ee.quaternion  # xyzw
        R_w_ee = R.from_quat(q_w_ee).as_matrix()
        T_w_ee = np.eye(4)
        T_w_ee[:3, :3] = R_w_ee
        T_w_ee[:3, 3] = t_w_ee
        point_w = T_w_ee * self.T_ee_c * point_c
        vec_w = R_w_ee * self.T_ee_c[:3, :3] * vec_c_3d
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
        T_w_ee_out = np.matmul(T_w_et, self.T_et_ee)
        p = {"R": T_w_ee_out[:3, :3], "t": T_w_ee_out[:3, 3]}
        self.goto_gripper(0.02)
        time.sleep(2)
        self.goto_pose(p, use_impedance=False)
        time.sleep(self.time_per_move)
    