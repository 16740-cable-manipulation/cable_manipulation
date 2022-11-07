import numpy as np
import cv2
import matplotlib.pyplot as plt
from locate_pixel_tmp import locatePixel





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

def processImage(image,targetColor):
    img = image
    img = cv2.resize(img,(800,640))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    blur_image = cv2.GaussianBlur(img, (3,3), 0)
    img_hsv = cv2.cvtColor(blur_image,cv2.COLOR_RGB2HSV)
    mask_oneColor = segmentFirstColor(img_hsv,targetColor)

    result_image = cv2.bitwise_and(img,img, mask=mask_oneColor)
    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    ax1.set_title('Target Cable Mask',fontdict = {'fontsize':8} )
    plt.imshow(mask_oneColor, cmap="gray")
    ax2 = fig.add_subplot(322)
    ax2.set_title('Target Cable',fontdict = {'fontsize':8} )

    plt.imshow(result_image)

    mask_allOther = segmentAllCableExceptOne(img_hsv,targetColor)
    # print(np.shape(mask_allOther)[0],np.shape(mask_allOther)[1])

    result_image2 = cv2.bitwise_and(img,img, mask=mask_allOther)
    ax3 = fig.add_subplot(323)
    ax3.set_title('Rest Cables Mask',fontdict = {'fontsize':8} )
   
    plt.imshow(mask_allOther, cmap="gray")
    ax4 = fig.add_subplot(324)
    ax4.set_title('Rest Cables',fontdict = {'fontsize':8} )
    # plt.subplot(3, 2, 1)
    plt.imshow(result_image2)


    LP = locatePixel(mask_oneColor,mask_allOther,50)
    boundaryBox = LP.iterateImage()
    # plt.subplot(3, 2, 5)
    ax5 = fig.add_subplot(325)
    plt.imshow(boundaryBox, cmap="gray")
    ax5.set_title('Possible Grab Pixels',fontdict = {'fontsize':8} )

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()
    ax5.set_axis_off()


    plt.show()
    




