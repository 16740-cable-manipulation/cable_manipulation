import cv2
import numpy as np
from time import time
img = cv2.imread("/Users/dhruvnaik/cable_manipulation/test_mask.png")
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

start,stop = (520,400), (565,445)
img2 = cv2.rectangle(img, start, stop, (255,0,0), 1)

window = img_grey[start[1]:stop[1], start[0]:stop[0]]

cv2.imshow("img",img2)
cv2.imshow("img22",img_grey)
cv2.waitKey(1000)

mesh_x, mesh_y = np.meshgrid(
    list(range(start[0],stop[0])), 
    list(range(start[1],stop[1]))
)

cv2.imshow('window',window)
cv2.imwrite("window.png",window)
cv2.waitKey(1000)
st = time()

contours, hierarchy  = cv2.findContours(window, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
end = time()
print(f"time taken: {end-st}")

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    x += start[0]
    y += start[1]

    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 1)
cv2.imshow('countours',img)
cv2.imwrite("countours.png",img)
mean_x = int(np.sum(mesh_x * window)/np.sum(window))
mean_y = int(np.sum(mesh_y * window)/np.sum(window))

img3 = cv2.circle(img, (mean_x,mean_y),4,(0,255,0),-1)
cv2.imwrite("mean_pix.png",img)

cv2.imshow('img_circle',img3)
cv2.waitKey(10000)
