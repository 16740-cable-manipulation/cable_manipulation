import numpy as np
import cv2
import matplotlib.pyplot as plt
from locate_pixel import locatePixel





def normalizeHSV(h,s,v):
    #opencv's hsv value are 179.255.255
    h_new = h/360*179 
    s_new = s*0.01*255
    v_new = v*0.01*255
    return (h_new,s_new,v_new)
    
r_low = normalizeHSV(0,60,30)
r_high = normalizeHSV(15,80,60)

g_low = normalizeHSV(165,75,20)
g_high = normalizeHSV(175,100,40)

b_low = normalizeHSV(220,35,35)
b_high = normalizeHSV(232,55,55)

y_low = normalizeHSV(45,40,45)
y_high = normalizeHSV(55,85,100)



def segmentFirstColor(input_image,color):
    #input_image needs to be in hsv
    #color is a string 
    if color == "red":
        mask_low = r_low
        mask_high = r_high
    elif color =="green":
        #GREEN - p bad
        mask_low = g_low
        mask_high = g_high
    elif color =="blue":
        #BLUE
        mask_low = b_low
        mask_high = b_high
    elif color =="yellow":
        #yellow
        mask_low = y_low
        mask_high = y_high

    mask = cv2.inRange(input_image, mask_low, mask_high)
    mask_blur = cv2.GaussianBlur(mask, (7,7), 0)

    return mask_blur

def segmentAllCableExceptOne(input_image,targetColor):
    #input_image needs to be in hsv
    #targetColor is a string that describes the color cable we want to move
    mask_r = cv2.inRange(input_image, r_low, r_high)
    mask_b = cv2.inRange(input_image, b_low, b_high)
    mask_g = cv2.inRange(input_image, g_low, g_high)
    mask_y = cv2.inRange(input_image, y_low, y_high)

    if targetColor == "red":    
        tmp1 = cv2.bitwise_or(mask_b,mask_g)
        tmp2 = cv2.bitwise_or(tmp1,mask_y)
        
    elif targetColor == "blue": 
        tmp1 = cv2.bitwise_or(mask_r,mask_g)
        tmp2 = cv2.bitwise_or(tmp1,mask_y)

    elif targetColor == "yellow": 
        tmp1 = cv2.bitwise_or(mask_b,mask_g)
        tmp2 = cv2.bitwise_or(tmp1,mask_r)
    elif targetColor == "green": 
        tmp1 = cv2.bitwise_or(mask_b,mask_r)
        tmp2 = cv2.bitwise_or(tmp1,mask_y)


    
    mask_blur = cv2.GaussianBlur(tmp2, (7,7), 0)
    return mask_blur

def processImage(inputImage,targetColor,visualize = False):

    img = cv2.cvtColor(inputImage,cv2.COLOR_BGR2RGB)

    blur_image = cv2.GaussianBlur(img, (3,3), 0)
    img_hsv = cv2.cvtColor(blur_image,cv2.COLOR_RGB2HSV)
    mask_oneColor = segmentFirstColor(img_hsv,targetColor)

    result_image = cv2.bitwise_and(img,img, mask=mask_oneColor)
    
    mask_allOther = segmentAllCableExceptOne(img_hsv,targetColor)
    # print(np.shape(mask_allOther)[0],np.shape(mask_allOther)[1])

    result_image2 = cv2.bitwise_and(img,img, mask=mask_allOther)
    


    LP = locatePixel(mask_oneColor,mask_allOther,55)
    mask_grabOK = LP.iterateImage()
    res = LP.findVector(mask_grabOK)
    if res is None:
        RuntimeError("Nothing is found")
    pt, vec = res
    ptprime = pt + vec
    pt = (int(pt[0]), int(pt[1]))
    ptprime = (int(ptprime[0]), int(ptprime[1]))
    print("pt,ptprime",pt,ptprime)
    vector = []

    if visualize == True:
        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        ax1.set_title('Target Cable Mask',fontdict = {'fontsize':8} )
        plt.imshow(mask_oneColor, cmap="gray")
        ax2 = fig.add_subplot(322)
        ax2.set_title('Target Cable',fontdict = {'fontsize':8} )
        plt.imshow(result_image)
        ax3 = fig.add_subplot(323)
        ax3.set_title('Rest Cables Mask',fontdict = {'fontsize':8} )
        plt.imshow(mask_allOther, cmap="gray")
        ax4 = fig.add_subplot(324)
        ax4.set_title('Rest Cables',fontdict = {'fontsize':8} )
        plt.imshow(result_image2)
        ax5 = fig.add_subplot(325)
        ax5.set_title('Possible Grab Pixels',fontdict = {'fontsize':8} )
      
        plt.imshow(mask_grabOK, cmap="gray")
        # X = vectorPairs[0]
        # Y = vectorPairs[1]

        ax6 = fig.add_subplot(326)
        mask_tmp = cv2.cvtColor(mask_grabOK.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        mask_with_vec = cv2.line(mask_tmp, pt, ptprime, (0, 255, 0), 4)
        # plt.imshow(mask_with_vec)
        cv2.imshow("image",mask_with_vec)
        cv2.waitKey(0)
        # plt.show()
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()
        ax4.set_axis_off()
        ax5.set_axis_off()
        ax6.set_axis_off()
       

        #plot vector
        # for vect in vectorPairs:
        #     p1 = vect[0]
        #     x1 = p1[0]
        #     y1 = p1[0]

        #     p2 = vect[1]
        #     x2 = p2[0]
        #     y2 = p2[1]
        #     ax6 = fig.add_subplot(326)
        #     plt.plot([x1,x2],[y1,y2]) 
        #     plt.show()
        # ax6 = fig.add_subplot(326)


    return (mask_grabOK,vector)


