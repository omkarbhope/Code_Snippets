import cv2 as cv
import math
from math import hypot
import random

# num = int(input("Enter a Number from 1 to 5"))
# num = str(num - 1)

def tshirt(frame,tshirt_image,center,L_Shoulder,R_Shoulder,L_hip,R_hip):
    pendant_image = cv.imread("tshirts/" + tshirt_image + ".png")
    width = int(int(hypot(L_Shoulder[0] - R_Shoulder[0],L_Shoulder[1] - R_Shoulder[1])) *1.8)
    height = int(int(hypot(L_Shoulder[0] - L_hip[0],L_Shoulder[1] - L_hip[1])) * 1.2)

    topLeft = ( int(center[0] - width / 2) , int( center[1] -height / 1.9) )
    bottomRight = ( int( center[0] + width / 2 ),int( center[1] +  height / 2))


    pendant = cv.resize(pendant_image, (int(width),int(height)),interpolation=cv.INTER_BITS)
    pendant_gray = cv.cvtColor(pendant, cv.COLOR_BGR2GRAY)
                
    _, pendant_mask = cv.threshold(pendant_gray, 5, 255, cv.THRESH_BINARY_INV)

    pendant_area = frame[topLeft[1]: topLeft[1] + int(height),topLeft[0]: topLeft[0] + int(width)]
    pendant_area_no_pendant = cv.bitwise_and(pendant_area, pendant_area, mask=pendant_mask)

    final_pendant = cv.add(pendant_area_no_pendant, pendant)

    frame[topLeft[1]: topLeft[1] + int(height),topLeft[0]: topLeft[0] + int(width)] = final_pendant

