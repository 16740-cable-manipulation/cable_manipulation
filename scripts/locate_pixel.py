import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import math

# given two masks: one with target color's cable, the other with the rest of the cables,
# find a location for a pixel of the targeted color is not clustered with other colors
# find a tangent line to the pixel


class locatePixel:
    def __init__(self, mask_targetColor, mask_restCable, boundary_size):
        self.mask_targetColor = (
            mask_targetColor  # np.where(mask_targetColor>0, 255, 0)
        )
        idx = np.argwhere(self.mask_targetColor > 0)
        self.mask_targetColor = np.zeros_like(
            self.mask_targetColor, dtype=np.uint8
        )
        self.mask_targetColor[idx[:, 0], idx[:, 1]] = 255
        # print(np.max(mask_targetColor))
        # cv2.imshow("img", self.mask_targetColor)
        # cv2.waitKey(0)
        self.mask_restCable = mask_restCable
        self.boundary_size = (
            boundary_size  # aka neighborhood size. needs to be odd
        )
        self.maskH = np.shape(self.mask_targetColor)[0]
        self.maskW = np.shape(self.mask_targetColor)[1]
        self.result = np.zeros((self.maskH, self.maskW))

    def isBoundaryEmptyOfOtherCables(self, cur_row, cur_col):
        init_r = min(
            max(0, cur_row - int((self.boundary_size - 1) / 2)), self.maskH - 1
        )
        init_c = min(
            max(0, cur_col - int((self.boundary_size - 1) / 2)), self.maskW - 1
        )
        neighborhood = self.mask_restCable[
            init_r : init_r + self.boundary_size,
            init_c : init_c + self.boundary_size,
        ]
        if len(neighborhood) > 0 and np.sum(neighborhood) > 0:
            return False
        return True

    def iterateImage(self):
        erosion_size = 5
        element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erosion_size + 1, 2 * erosion_size + 1),
            (erosion_size, erosion_size),
        )

        self.mask_targetColor = cv2.erode(self.mask_targetColor, element)

        # self.mask_targetColor = np.where(self.mask_targetColor>0, 250, 0)
        idx = np.argwhere(self.mask_targetColor > 0)
        # self.mask_targetColor = np.zeros_like(self.mask_targetColor, dtype=np.uint8)
        # self.mask_targetColor[idx[:,0], idx[:,1]] = 255
        # print(idx.shape[0])
        # cv2.imshow("img",self.mask_targetColor )
        # cv2.waitKey(0)

        for id in idx:
            # print(i)
            r = id[0]
            c = id[1]
            if self.isBoundaryEmptyOfOtherCables(r, c):
                self.result[r, c] = 255

        # final = cv2.bitwise_or(self.mask_targetColor,self.mask_targetColor,mask = self.result)
        return self.result

    def calcDistance(self, x1, y1, x2, y2):
        result = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        return result
