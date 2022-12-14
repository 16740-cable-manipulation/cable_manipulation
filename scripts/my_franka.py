from math import atan2
import time
from tkinter.messagebox import NO
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import yaml
from scipy.spatial.transform import Rotation as R
import copy
from action import Action

GRIPPER_GRASP = 0
GRIPPER_UNGRASP = 1
GRIPPER_NONE = 2

ANGLE_WRAP = 0
ANGLE_CLIP = 1


class MyFranka:
    def __init__(self):
        self.time_per_move = 2
        self.small_z_offset = -0.028
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

    def open_gripper(self):
        self.fa.goto_gripper(0.05)
        time.sleep(1)

    def close_gripper(self):
        self.fa.close_gripper()
        time.sleep(1.5)

    def reset_joint_and_gripper(self):
        self.fa.open_gripper()
        time.sleep(2)
        self.fa.reset_joints()
        time.sleep(self.time_per_move)

    def goto_pose(self, pose, sleep=2, use_impedance=False):
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
        time.sleep(sleep)

    def goto_capture_pose(self):
        home_pose = {
            "R": np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            ),
            "t": np.array([0.395, 0.0, 0.57]),
        }
        self.goto_pose(home_pose, sleep=3.5, use_impedance=False)

    def goto_middle_pose(self):
        home_pose = {
            "R": np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            ),
            "t": np.array([0.395, 0.0, 0.25]),
        }
        self.goto_pose(home_pose, sleep=3.5, use_impedance=False)

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

    def get_gripper_width(self):
        gripper_width = self.fa.get_gripper_width()

        return gripper_width

    def go_to_vec(self, theta):
        tf_w_ee = self.get_pose()
        T_w_et = np.eye(4)
        T_w_et[:3, 3] = copy.deepcopy(tf_w_ee["t"])
        T_w_et[:3, :3] = tf_w_ee["R"]
        r_ee0_ee1 = R.from_euler("zyx", [theta, 0, 0]).as_matrix()
        dT = np.eye(4)
        dT[:3, :3] = r_ee0_ee1
        T_w_et = np.matmul(T_w_et, dT)
        home_pose = {
            "R": T_w_et[:3, :3],
            "t": T_w_et[:3, 3],
        }
        self.goto_pose(home_pose, sleep=self.time_per_move, use_impedance=False)

    def goto_point_and_vec(
        self,
        point_c,
        vec_c_3d,
        tf_w_ee=None,
        manipulate_gripper=GRIPPER_NONE,
        angle_mode=ANGLE_WRAP,
        change_angle=False,
    ):
        """``tf_w_ee``: if pass in a pose dict, the function will use this as reference
        instead of the current ee and camera pose in world frame

        ``change_angle``: if true, the pointing direction will be the vector direction + PI.
        Then it will further be wrapped or clipped, depending on ``angle_mode``"""
        # transform point to world frame
        if tf_w_ee is None:  # use the current ee and camera pose as reference
            tf_w_ee = self.get_pose()  # actually is tf_w_et
        T_w_ee = np.eye(4)
        T_w_ee[:3, :3] = copy.deepcopy(tf_w_ee["R"])
        T_w_ee[:3, 3] = copy.deepcopy(tf_w_ee["t"])
        # print("translation: ", tf_w_ee["t"])
        point_c_homo = np.reshape(np.hstack((point_c, np.ones(1))), (-1, 1))
        # print(point_c_homo)
        p_et = np.matmul(self.T_ee_c, point_c_homo)
        # print(p_et)
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
        if change_angle is True:  # flip vector direction
            angle += np.pi
            if angle > np.pi:
                angle -= 2 * np.pi
        angle_changed = False
        if angle_mode == ANGLE_WRAP:
            if angle > np.pi / 2:
                angle -= np.pi
                angle_changed = True
            elif angle < -np.pi / 2:
                angle += np.pi
                angle_changed = True
        elif angle_mode == ANGLE_CLIP:
            angle = np.clip(angle, -np.pi / 2 + 0.05, np.pi / 2 - 0.05)
            angle_changed = True
        r_ee0_et = R.from_euler("zyx", [angle, 0, 0])
        R_w_et = np.matmul(r_w_ee0.as_matrix(), r_ee0_et.as_matrix())
        T_w_et[:3, :3] = R_w_et
        p = {"R": T_w_et[:3, :3], "t": T_w_et[:3, 3]}
        print("T_w_et: ")
        print(T_w_et)

        if manipulate_gripper == GRIPPER_GRASP:
            num_attemp = 0
            while num_attemp < 5:
                self.open_gripper()
                self.goto_pose(p, sleep=self.time_per_move, use_impedance=False)
                self.close_gripper()
                gripper_width = self.get_gripper_width()
                if gripper_width > 0.002:
                    break
                print("REGRASP!!")
                p["t"] += [0, 0, -0.006]
                num_attemp += 1

        elif manipulate_gripper == GRIPPER_UNGRASP:
            self.goto_pose(p, sleep=self.time_per_move, use_impedance=False)
            # push surrounding cables
            self.fa.open_gripper()
            time.sleep(1.5)
            self.open_gripper()
        else:
            self.goto_pose(p, sleep=self.time_per_move, use_impedance=False)
        return angle_changed

    def exe_action(self, action: Action):
        # assume we're at capture pose
        tf_w_ee = self.get_pose()
        # grasp action.pick_3d
        angle_changed = self.goto_point_and_vec(
            action.pick_3d,
            action.pick_vec_3d,
            tf_w_ee=tf_w_ee,
            manipulate_gripper=GRIPPER_GRASP,
            angle_mode=ANGLE_WRAP,
        )
        # lift to action.z
        curr_pose = self.get_pose()
        self.goto_pose(
            {
                "R": curr_pose["R"],
                "t": curr_pose["t"] + [0, 0, action.z],
            }
        )
        # move to a point above action.place_3d, with height z
        print("place_3d: ", action.place_3d)
        self.goto_point_and_vec(
            action.place_3d + [0, 0, -action.z * 0.7],
            action.place_vec_3d,
            tf_w_ee=tf_w_ee,
            manipulate_gripper=GRIPPER_NONE,
            change_angle=angle_changed,
            angle_mode=ANGLE_CLIP,
        )
        # lower to table and ungrasp
        self.goto_point_and_vec(
            action.place_3d,
            action.place_vec_3d,
            tf_w_ee=tf_w_ee,
            manipulate_gripper=GRIPPER_UNGRASP,
            change_angle=angle_changed,
            angle_mode=ANGLE_CLIP,
        )


if __name__ == "__main__":
    fa = MyFranka()
    fa.reset_joint_and_gripper()
    fa.close_gripper()
    # fa.go_to_vec(np.pi * 2 / 3)
