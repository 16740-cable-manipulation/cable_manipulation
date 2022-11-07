#test file for cable_segmentation
import cable_segmentation
import sys

if __name__ == "__main__":
    # inputImage = sys.argv[1]
    # color = sys.argv[2]
    inputImage = "d:/XinyuWang/2022_Fall/16740/cable_manipulation/cableImages/cableBundle2.jpg"
    color = "blue"
    cable_segmentation.processImage(inputImage,color,visualize=True)

