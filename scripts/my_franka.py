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
        self.T_et_ee = np.eye(4)
        self.T_ee_c = self.load_extrinsic()

    def load_extrinsic(self):
        # transformation from fingertip of ee to center of ee
        T_et_ee = np.eye(4)
        T_et_ee[3, 3] = -0.1149  # only z translation
        # Step 0: load T_ee_c
        T_ee_c = np.eye(4)
        with open("cable_manipulation/test.yaml") as f:
            loaded = yaml.load(f, Loader=yaml.Loader)
            R_ee_c = R.from_quat(np.array(loaded["rot_e_c"]))
            T_ee_c[:3, :3] = R_ee_c.as_matrix()
            T_ee_c[:3, 3] = np.array(loaded["trans_e_c"])
            print(T_ee_c)
