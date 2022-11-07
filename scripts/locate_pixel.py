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
    
    def isBoundaryEmptyOfOtherCables(self,allOtherCableMask,cur_row,cur_col):
        for row in range(self.boundary_size):
            for col in range(self.boundary_size):
                
                    if allOtherCableMask[cur_row+row,cur_col+col] !=0:
                        # print("All cable appeared")
                        return False
                #     else:
                #         print("only target cable appeared")
                #         pass
                # else:
                #     print("no target cable appeared")
                #     pass
                
                    
            
        
        return True
    
    def iterateImage(self):
        for r in range(0,self.maskW-self.boundary_size):
            for c in range(0,self.maskH-self.boundary_size):
                if self.mask_targetColor[r,c] != 0 :

                    if self.isBoundaryEmptyOfOtherCables(self.mask_restCable,r,c):

                        self.result[r:r+self.boundary_size,c:c+self.boundary_size] = 1
                    else:
                        self.result[r:r+self.boundary_size,c:c+self.boundary_size] = 0
                    # print("BBBB itearate through one boundary box",r,c)

        # final = cv2.bitwise_or(self.mask_targetColor,self.mask_targetColor,mask = self.result)
        
        return self.result






# if __name__ =="__main__":
#     mask_targetColor = #sys.argv[1]
#     mask_restCable = #sys.argv[2]
#     boundary_size = 70#sys.argv[3]
    
