import numpy as np
import cv2


class CableSegmentation:
    def __init__(self):
        # segmentation thresholds for iphone images
        # self.r_low = self.normalizeHSV(0, 60, 30)
        # self.r_high = self.normalizeHSV(15, 80, 60)

        # self.g_low = self.normalizeHSV(165, 75, 20)
        # self.g_high = self.normalizeHSV(175, 100, 40)

        # self.b_low = self.normalizeHSV(220, 35, 35)
        # self.b_high = self.normalizeHSV(232, 55, 55)

        # self.y_low = self.normalizeHSV(45, 40, 45)
        # self.y_high = self.normalizeHSV(55, 85, 100)

        # segmentation thresholds for realsense images
        self.r_low = self.normalizeHSV(0, 35, 25)
        self.r_high = self.normalizeHSV(20, 90, 50)

        self.g_low = self.normalizeHSV(165, 75, 20)
        self.g_high = self.normalizeHSV(175, 100, 40)

        self.b_low = self.normalizeHSV(205, 40, 20)
        self.b_high = self.normalizeHSV(225, 75, 50)

        self.y_low = self.normalizeHSV(35, 40, 25)
        self.y_high = self.normalizeHSV(50, 88, 70)

    def normalizeHSV(self, h, s, v):
        # opencv's hsv value are 179.255.255
        h_new = h / 360 * 179
        s_new = s * 0.01 * 255
        v_new = v * 0.01 * 255
        return (h_new, s_new, v_new)

    def segmentFirstColor(self, input_image, color):
        # input_image needs to be in hsv
        # color is a string
        if color == "red":
            mask_low = self.r_low
            mask_high = self.r_high
        elif color == "green":
            # GREEN - p bad
            mask_low = self.g_low
            mask_high = self.g_high
        elif color == "blue":
            # BLUE
            mask_low = self.b_low
            mask_high = self.b_high
        elif color == "yellow":
            # yellow
            mask_low = self.y_low
            mask_high = self.y_high

        mask = cv2.inRange(input_image, mask_low, mask_high)
        mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)

        return mask_blur

    def segmentAllCableExceptOne(self, input_image, targetColor):
        # input_image needs to be in hsv
        # targetColor is a string that describes the color cable we want to move
        mask_r = cv2.inRange(input_image, self.r_low, self.r_high)
        mask_b = cv2.inRange(input_image, self.b_low, self.b_high)
        mask_g = cv2.inRange(input_image, self.g_low, self.g_high)
        mask_y = cv2.inRange(input_image, self.y_low, self.y_high)

        if targetColor == "red":
            tmp1 = cv2.bitwise_or(mask_b, mask_g)
            tmp2 = cv2.bitwise_or(tmp1, mask_y)

        elif targetColor == "blue":
            tmp1 = cv2.bitwise_or(mask_r, mask_g)
            tmp2 = cv2.bitwise_or(tmp1, mask_y)

        elif targetColor == "yellow":
            tmp1 = cv2.bitwise_or(mask_b, mask_g)
            tmp2 = cv2.bitwise_or(tmp1, mask_r)
        elif targetColor == "green":
            tmp1 = cv2.bitwise_or(mask_b, mask_r)
            tmp2 = cv2.bitwise_or(tmp1, mask_y)

        mask_blur = cv2.GaussianBlur(tmp2, (7, 7), 0)
        return mask_blur
