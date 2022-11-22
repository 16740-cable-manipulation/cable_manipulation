# from cable_manipulation import CableManipulation
import sys
import cv2
import numpy as np
import math


class Discretize:
    def __init__(self, cableMask):
        self.windowSize = 45
        self.cableMask = cableMask
        self.maskH = np.shape(self.cableMask)[0]
        self.maskW = np.shape(self.cableMask)[1]
        idx = np.argwhere(self.cableMask > 0)
        self.startPosition = idx[0]
        self.resultPixel = []

    def findMeanPixel(self, neighborhood):
        n_shape = neighborhood.shape
        if len(n_shape) == 3:
            neighborhood = neighborhood.mean(-1)
        mesh_x, mesh_y = np.meshgrid(
            list(range(len(neighborhood))), list(range(len(neighborhood[0])))
        )
        mean_x = int(np.sum(mesh_x * neighborhood) / np.sum(neighborhood))
        mean_y = int(np.sum(mesh_y * neighborhood) / np.sum(neighborhood))
        # Need to add X,Y offsets after values are returned
        return (mean_x, mean_y)

    def findTangent(self, neighborhood):
        cur_idx = np.argwhere(neighborhood > 0)
        mid = math.floor(len(cur_idx) / 2)
        offset = 5  # distance between two points
        p1 = cur_idx[mid]
        p2 = cur_idx[mid + offset]
        tangentVector = [p2[1] - p1[1], p2[0] - p1[0]]
        return tangentVector

    def findContours(self, neighborhood):
        contours, hierarchy  = cv2.findContours(neighborhood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return len(contours)

    def isEndCable(self, neighborhood):
        # Assumes binary mask
        return neighborhood.sum() == 0

    def slideWindow(self):
        # given point (r,c) find mean_r, mean_c in neighborhood, store mean_r, mean_c
        # find tangentVector in neighbohood
        # move to next neighborhood based on the tangentVector
        cur_row = self.startPosition[0]
        cur_col = self.startPosition[1]
        init_r = min(
            max(0, cur_row - int((self.windowSize - 1) / 2)), self.maskH - 1
        )
        init_c = min(
            max(0, cur_col - int((self.windowSize - 1) / 2)), self.maskW - 1
        )
        neighborhood = self.cableMask[
            init_r : init_r + self.windowSize, init_c : init_c + self.windowSize
        ]
        meanPixel = self.findMeanPixel(neighborhood)
        self.resultPixel.append(meanPixel)
        tangentVector = self.findTangent(neighborhood)

        nextPoint = np.array([init_r, init_c]) + tangentVector

        return

    # TODO
    def getCablesDataFromImage(self, img):
        """Generate cable data dictionary given an cv BGR image"""
        data1 = {"coords": None, "pos": None, "cx": None, "color": "red"}
        data2 = {"coords": None, "pos": None, "cx": None, "color": "blue"}
        cables_data = {"cableID1": data1, "cableID2": data2}
        return cables_data

