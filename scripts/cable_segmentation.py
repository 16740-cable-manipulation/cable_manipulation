import numpy as np
import cv2
import matplotlib.pyplot as plt






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


#hsv threshold 
#RED - p good
def segmentFirstColor(input_image,color):
    #input_image needs to be in hsv
    #color is a string 
    if color == "red":
        mask_low = normalizeHSV(0,60,30)
        mask_high = normalizeHSV(15,80,60)
    elif color =="green":
        #GREEN - p bad
        mask_low = normalizeHSV(165,75,20)
        mask_high = normalizeHSV(175,100,40)
    elif color =="blue":
        #BLUE
        mask_low = normalizeHSV(220,35,35)
        mask_high = normalizeHSV(232,55,55)
    elif color =="yellow":
        #yellow
        mask_low = normalizeHSV(45,40,45)
        mask_high = normalizeHSV(55,85,100)





    mask = cv2.inRange(input_image, mask_low, mask_high)
    mask_blur = cv2.GaussianBlur(mask, (7,7), 0)

    return mask_blur

def segmentAllCableExceptOne(input_image,targetColor):
    #input_image needs to be in hsv
    #targetColor is a string that describes the color cable we want to move


    if targetColor == "red":
        low1 = cv2.bitwise_or(g_low,b_low)
        low2 = cv2.bitwise_or(low1,y_low)
        high1 = cv2.bitwise_or(g_high,b_high)
        high2 = cv2.bitwise_or(high1,y_high)

        mask = cv2.inRange(input_image, low2, high2)

    elif targetColor == "blue": 
        low1 = cv2.bitwise_or(g_low,r_low)
        low2 = cv2.bitwise_or(low1,y_low)
        high1 = cv2.bitwise_or(g_high,r_high)
        high2 = cv2.bitwise_or(high1,y_high)
    elif targetColor == "yellow": 
        low1 = cv2.bitwise_or(g_low,b_low)
        low2 = cv2.bitwise_or(low1,r_low)
        high1 = cv2.bitwise_or(g_high,b_high)
        high2 = cv2.bitwise_or(high1,r_high)
    elif targetColor == "green": 
        low1 = cv2.bitwise_or(r_low,b_low)
        low2 = cv2.bitwise_or(low1,y_low)
        high1 = cv2.bitwise_or(r_high,b_high)
        high2 = cv2.bitwise_or(high1,y_high)


    
    mask_blur = cv2.GaussianBlur(mask, (7,7), 0)
    return mask_blur

if __name__ =="__main__":
    img = cv2.imread("cableImages/cableBundle.jpg")
    img = cv2.resize(img,(800,640))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    blur_image = cv2.GaussianBlur(img, (3,3), 0)
    plt.imshow(blur_image)
    plt.show()

    img_hsv = cv2.cvtColor(blur_image,cv2.COLOR_RGB2HSV)

    mask_oneColor = segmentFirstColor(img_hsv,"red")
    result = cv2.bitwise_and(img,img, mask=mask_oneColor)
    plt.subplot(2, 2, 1)
    plt.imshow(mask_oneColor, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(result)

    mask_allOther = segmentAllCableExceptOne(img_hsv,"red")
    result = cv2.bitwise_and(img,img, mask=mask_allOther)
    plt.subplot(2, 2, 3)
    plt.imshow(mask_allOther, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(result)


    plt.show()
