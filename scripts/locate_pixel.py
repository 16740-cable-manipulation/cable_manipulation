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
        self.boundary_size = boundary_size  # needs to be odd
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

    def findVector(self, _mask_grabOK):
        # mask_grabOk = self.iterateImage()
        dw = 100
        dh = 100
        mask_grabOK = _mask_grabOK[dh : self.maskH - dh, dw : self.maskW - dw]
        idx = np.argwhere(mask_grabOK > 0)
        # print("number of pixel",len(idx))
        # quit()
        gridSize = 40

        # for id in idx:
        id = np.random.choice(np.arange(idx.shape[0]))
        r = idx[id, 0]
        c = idx[id, 1]

        init_r = min(max(0, r - int((gridSize - 1) / 2)), self.maskH - 1)
        init_c = min(max(0, c - int((gridSize - 1) / 2)), self.maskW - 1)
        neighborhood = mask_grabOK[
            init_r : init_r + gridSize, init_c : init_c + gridSize
        ]
        idx_neighborhood = np.argwhere(neighborhood > 0)
        # if there is more than 1 pixel in this neighbohood and the distance between these neighboods are over a threshold value
        # out the pixel pair in a list
        p1 = None
        p1_idx = 0
        p2 = None
        if len(idx_neighborhood) < 2:
            return None
        print(len(idx_neighborhood))

        edgeA = np.vstack((np.zeros(gridSize), np.arange(gridSize))).T
        edgeB = np.vstack(
            (np.arange(1, gridSize), np.ones(gridSize - 1) * (gridSize - 1))
        ).T
        edgeC = np.vstack(
            (
                np.ones(gridSize - 1) * (gridSize - 1),
                np.arange(gridSize - 2, -1, -1),
            )
        ).T
        edgeD = np.vstack(
            (np.arange(gridSize - 2, 0, -1), np.zeros(gridSize - 2))
        ).T
        edges = np.vstack((edgeA, edgeB, edgeC, edgeD))
        for idx in range(len(edges)):
            tmp_r = int(edges[idx, 0])
            tmp_c = int(edges[idx, 1])
            if neighborhood[tmp_r, tmp_c] > 0:
                if p1 is None:
                    p1 = np.array([tmp_r, tmp_c])
                    p1_idx = idx
                else:
                    if idx - p1_idx > gridSize:
                        p2 = np.array([tmp_r, tmp_c])
        if p1 is None or p2 is None:
            return None
        vec = np.floor(np.flip((p2 - p1)) * 3)
        pt = np.array([c + dw, r + dh])
        ptprime = pt + vec
        if (
            ptprime[0] < 0
            or ptprime[0] >= self.maskW
            or ptprime[1] < 0
            or ptprime[1] >= self.maskH
        ):
            vec = -vec
        pt = (int(pt[0]), int(pt[1]))
        vec = (int(vec[0]), int(vec[1]))
        return pt, vec
