import cv2
import numpy as np
import copy
import math

#Parameters
global cap_region_x_begin, cap_region_y_end, threshold, blurValue, bgSubThreshold, learningRate
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

#Variables
global isBgCaptured, triggerSwitch
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

# Instanciantions
global bgModel
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=3)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

img = removeBG(cv2.imread('test.png'))
#cv2.imshow('teste img', cv2.imread('test.png'))
   
# convert the image into binary image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
#cv2.imshow('blur', blur)
ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
cv2.imwrite('test_binarizeds.png',thresh.copy())




