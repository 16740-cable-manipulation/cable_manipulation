#test file for cable_segmentation
import cable_segmentation
import sys
import cv2
if __name__ == "__main__":
    # inputImage = sys.argv[1]
    # color = sys.argv[2]
    inputImage = cv2.imread("d:/XinyuWang/2022_Fall/16740/cable_manipulation/cableImages/cableBundle2.jpg")
    inputImage = cv2.resize(inputImage,(800,640))
    color = "blue"
    cable_segmentation.processImage(inputImage,color,visualize=False)

