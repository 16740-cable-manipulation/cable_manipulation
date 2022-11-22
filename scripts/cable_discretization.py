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
        self.rim_offset = (
            100  # for cropping the center area before selecting grasp point
        )
        self.vector_grid_dize = 41  # for computing the vector

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

    # def findTangent(self, neighborhood):
    #     cur_idx = np.argwhere(neighborhood > 0)
    #     mid = math.floor(len(cur_idx) / 2)
    #     offset = 5  # distance between two points
    #     p1 = cur_idx[mid]
    #     p2 = cur_idx[mid + offset]
    #     tangentVector = [p2[1] - p1[1], p2[0] - p1[0]]
    #     return tangentVector

    def findVector(self, mask_grabOK_orig):
        gridSize = self.vector_grid_dize
        dh = self.rim_offset
        dw = self.rim_offset
        mask_grabOK = mask_grabOK_orig[
            dh : self.maskH - dh, dw : self.maskW - dw
        ]
        idx = np.argwhere(mask_grabOK > 0)
        id = np.random.choice(np.arange(idx.shape[0]))
        r = idx[id, 0]  # row coordinate in the cropped mask
        c = idx[id, 1]
       

        init_r = min(
            max(0, r - int((gridSize - 1) / 2)),
            self.maskH - 1,
        )
        init_c = min(
            max(0, c - int((gridSize - 1) / 2)),
            self.maskW - 1,
        )
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
        vec = np.floor(np.flip((p2 - p1)))

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

        meanPixel = self.findMeanPixel(neighborhood)
        return pt, vec, meanPixel
    
    def visualize_vector(self, mask_grabOK, pt, ptprime):
        mask_tmp = cv2.cvtColor(
            mask_grabOK.astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        mask_with_vec = cv2.line(mask_tmp, pt, ptprime, (0, 255, 0), 4)
        # plt.imshow(mask_with_vec)
        cv2.imshow("image", mask_with_vec)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        # cur_row = self.startPosition[0]
        # cur_col = self.startPosition[1]
        # init_r = min(
        #     max(0, cur_row - int((self.windowSize - 1) / 2)), self.maskH - 1
        # )
        # init_c = min(
        #     max(0, cur_col - int((self.windowSize - 1) / 2)), self.maskW - 1
        # )
        # neighborhood = self.cableMask[
        #     init_r : init_r + self.windowSize, init_c : init_c + self.windowSize
        # ]
        # meanPixel = self.findMeanPixel(neighborhood)
        # self.resultPixel.append(meanPixel)
        # tangentVector = self.findTangent(neighborhood)

        # nextPoint = np.array([init_r, init_c]) + tangentVector
        while not self.isEndCable():
            

        return

    # TODO
    def getCablesDataFromImage(self, img):
        """Generate cable data dictionary given an cv BGR image"""
        data1 = {"coords": None, "pos": None, "cx": None, "color": "red"}
        data2 = {"coords": None, "pos": None, "cx": None, "color": "blue"}
        cables_data = {"cableID1": data1, "cableID2": data2}
        return cables_data

