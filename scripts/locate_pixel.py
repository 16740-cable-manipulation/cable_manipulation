import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


#given two masks: one with target color's cable, the other with the rest of the cables, 
#find a location for a pixel of the targeted color is not clustered with other colors
#find a tangent line to the pixel

class locatePixel():
    def __init__(self,mask_targetColor,mask_restCable,boundary_size):
        self.mask_targetColor = mask_targetColor
        self.mask_restCable = mask_restCable
        self.boundary_size = boundary_size
        self.maskW = np.shape(self.mask_targetColor)[0]
        self.maskH = np.shape(self.mask_targetColor)[1]
        self.result = np.zeros((self.maskW,self.maskH))
    
    def isBoundaryEmpty(self,given_mask,cur_row,cur_col):
        for row in range(self.boundary_size):
            for col in range(self.boundary_size):
                if given_mask[cur_row+row,cur_col+col] == 0:
                    return True
                else:
                    return False
    
    def iterateImage(self):
        for r in range(self.maskW-self.boundary_size):
            for c in range(self.maskH-self.boundary_size):
                if not self.isBoundaryEmpty(self.mask_targetColor,r,c):
                    #if targetCable is found, ie targetCable is not in boundaryBox
                    if self.isBoundaryEmpty(self.mask_restCable,r,c):
                        #and if this area is free of other cables
                        self.result[r,c] = 255
        
        return self.result






# if __name__ =="__main__":
#     mask_targetColor = #sys.argv[1]
#     mask_restCable = #sys.argv[2]
#     boundary_size = 70#sys.argv[3]
    
