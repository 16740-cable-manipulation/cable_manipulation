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
    
    def isBoundaryEmptyOfOtherCables(self,targetColorMask,allOtherCableMask,cur_row,cur_col):
        for row in range(self.boundary_size):
            for col in range(self.boundary_size):
                if targetColorMask[cur_row+row,cur_col+col] !=0:
                    if allOtherCableMask[cur_row+row,cur_col+col] !=0:
                        print("All cable appeared")
                        return False
                #     else:
                #         print("only target cable appeared")
                #         pass
                # else:
                #     print("no target cable appeared")
                #     pass
                
                    
            
        
        return True
    
    def iterateImage(self):
        for r in range(0,self.maskW-self.boundary_size,self.boundary_size):
            for c in range(0,self.maskH-self.boundary_size,self.boundary_size):
                if self.isBoundaryEmptyOfOtherCables(self.mask_targetColor,self.mask_restCable,r,c):
                    #if targetCable is found, ie targetCable is not in boundaryBox
            
                        #and if this area is free of other cables
                        # for box_w in range(self.boundary_size):
                        #     for box_h in range(self.boundary_size):
                                # self.result[r+box_w,c+box_h] = 255
                        
                        # print("AAAA itearate through one boundary box",r,c)
                        self.result[r:r+self.boundary_size,c:c+self.boundary_size] = 255
                else:
                    self.result[r:r+self.boundary_size,c:c+self.boundary_size] = 0
                    # print("BBBB itearate through one boundary box",r,c)


        return self.result






# if __name__ =="__main__":
#     mask_targetColor = #sys.argv[1]
#     mask_restCable = #sys.argv[2]
#     boundary_size = 70#sys.argv[3]
    
