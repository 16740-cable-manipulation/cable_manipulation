import imp
from random import sample
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rs_driver import Realsense
from cable_segmentation import CableSegmentation
from locate_pixel import locatePixel


class CableManipulation:
    def __init__(self, w, h, use_rs=False):
        self.seg = CableSegmentation()
        self.use_rs = use_rs
        self.image_width = w
        self.image_height = h
        self.rim_offset = (
            100  # for cropping the center area before selecting grasp point
        )
        self.vector_grid_dize = 40  # for computing the vector
        if self.use_rs is True:
            self.realsense = Realsense()

    def preprocessImage(self, inputImage, targetColor):
        img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)

        blur_image = cv2.GaussianBlur(img, (3, 3), 0)
        img_hsv = cv2.cvtColor(blur_image, cv2.COLOR_RGB2HSV)
        mask_oneColor = self.seg.segmentFirstColor(img_hsv, targetColor)
        result_image = cv2.bitwise_and(img, img, mask=mask_oneColor)

        mask_allOther = self.seg.segmentAllCableExceptOne(img_hsv, targetColor)
        result_image2 = cv2.bitwise_and(img, img, mask=mask_allOther)

        return mask_oneColor, mask_allOther, result_image, result_image2

    def processImage(
        self, inputImage, targetColor, inputDepth=None, visualize=False
    ):
        (
            mask_oneColor,
            mask_allOther,
            result_image,
            result_image2,
        ) = self.preprocessImage(inputImage, targetColor)

        LP = locatePixel(mask_oneColor, mask_allOther, 55)
        mask_grabOK = LP.iterateImage()
        res = self.findVector(mask_grabOK, inputDepth)
        if res is None:
            RuntimeError("Nothing is found")
        pt, vec = res
        ptprime = (pt[0] + vec[0], pt[1] + vec[1])
        print("pt,ptprime", pt, ptprime)

        if visualize == True:
            self.visualize_processed_images(
                mask_oneColor,
                mask_allOther,
                result_image,
                result_image2,
                mask_grabOK,
            )
            self.visualize_vector(mask_grabOK, pt, ptprime)

        return (mask_grabOK, pt, vec)

    def findVector(self, _mask_grabOK, depth):
        gridSize = self.vector_grid_dize
        dh = self.rim_offset
        dw = self.rim_offset
        mask_grabOK = _mask_grabOK[
            dh : self.image_height - dh, dw : self.image_width - dw
        ]
        idx = np.argwhere(mask_grabOK > 0)

        # TODO randomly find pixel with valid depth
        if depth is None:
            id = np.random.choice(np.arange(idx.shape[0]))
            r = idx[id, 0]  # row coordinate in the cropped mask
            c = idx[id, 1]
            pt = np.array([c + dw, r + dh])
        else:
            sample = 0
            r = None
            c = None
            pt = None
            while sample < 50:
                id = np.random.choice(np.arange(idx.shape[0]))
                r = idx[id, 0]  # row coordinate in the cropped mask
                c = idx[id, 1]
                pt = np.array(
                    [c + dw, r + dh]
                )  # pixel coordinate in the original mask
                # check if it has valid depth
                point_c = self.realsense.deproject_pixel(depth, pt[0], pt[1])
                if point_c[2] >= 0.05 or point_c[2] <= 1:
                    break
                sample += 1
            if r is None:
                RuntimeError("Sampled 50 pixels, none has valid depth!")
                if self.use_rs is True:
                    self.realsense.close()
                quit(1)

        init_r = min(max(0, r - int((gridSize - 1) / 2)), self.image_height - 1)
        init_c = min(max(0, c - int((gridSize - 1) / 2)), self.image_width - 1)
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

        ptprime = pt + vec
        if (
            ptprime[0] < 0
            or ptprime[0] >= self.image_width
            or ptprime[1] < 0
            or ptprime[1] >= self.image_height
        ):
            vec = -vec
        pt = (int(pt[0]), int(pt[1]))
        vec = (int(vec[0]), int(vec[1]))
        return pt, vec

    def visualize_vector(self, mask_grabOK, pt, ptprime):
        mask_tmp = cv2.cvtColor(
            mask_grabOK.astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        mask_with_vec = cv2.line(mask_tmp, pt, ptprime, (0, 255, 0), 4)
        # plt.imshow(mask_with_vec)
        cv2.imshow("image", mask_with_vec)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_processed_images(
        self,
        mask_oneColor,
        mask_allOther,
        result_image,
        result_image2,
        mask_grabOK,
    ):
        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        ax1.set_title("Target Cable Mask", fontdict={"fontsize": 8})
        plt.imshow(mask_oneColor, cmap="gray")
        ax2 = fig.add_subplot(322)
        ax2.set_title("Target Cable", fontdict={"fontsize": 8})
        plt.imshow(result_image)
        ax3 = fig.add_subplot(323)
        ax3.set_title("Rest Cables Mask", fontdict={"fontsize": 8})
        plt.imshow(mask_allOther, cmap="gray")
        ax4 = fig.add_subplot(324)
        ax4.set_title("Rest Cables", fontdict={"fontsize": 8})
        plt.imshow(result_image2)
        ax5 = fig.add_subplot(325)
        ax5.set_title("Possible Grab Pixels", fontdict={"fontsize": 8})

        plt.imshow(mask_grabOK, cmap="gray")
        # X = vectorPairs[0]
        # Y = vectorPairs[1]
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()
        ax4.set_axis_off()
        ax5.set_axis_off()
        plt.show()
        plt.close()