import imp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rs_driver import Realsense
from cable_segmentation import CableSegmentation
from locate_pixel import locatePixel


class CableManipulation:
    def __init__(self, use_rs=False):
        self.seg = CableSegmentation()
        if use_rs is True:
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

    def processImage(self, inputImage, targetColor, visualize=False):
        (
            mask_oneColor,
            mask_allOther,
            result_image,
            result_image2,
        ) = self.preprocessImage(inputImage, targetColor)

        LP = locatePixel(mask_oneColor, mask_allOther, 55)
        mask_grabOK = LP.iterateImage()
        res = LP.findVector(mask_grabOK)
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

    def visualize_vector(self, mask_grabOK, pt, ptprime):
        mask_tmp = cv2.cvtColor(
            mask_grabOK.astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        mask_with_vec = cv2.line(mask_tmp, pt, ptprime, (0, 255, 0), 4)
        # plt.imshow(mask_with_vec)
        cv2.imshow("image", mask_with_vec)
        cv2.waitKey(0)

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
