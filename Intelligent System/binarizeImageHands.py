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
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# Inicializations
#D = 300 # Dimension to reshape images, when vectorize one vector -> len = 90000 (300x300)

def binarize_set_hand_image_file(Filename,numberOfHandsType):
    # Number of Persons in DataBase
    N = numberOfHandsType

    # Number of image for each type hand in Database
    Ni = 15
    
    # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = 'hand'
    str_2 = '.pgm'
    
    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes

            if i < 14:
                str_ = '{}/{}_{}_{}{}'.format(filename_str, str_1, i + 1, j + 1, str_2)
                img_file = cv2.imread(str_,0)
                img_withoutBG = removeBG(img_file)
                blur = cv2.GaussianBlur(img_withoutBG, (blurValue, blurValue), 0)
                cv2.imwrite('hands_blur_marcel_dataset/{}_{}_{}_blur{}'.format(str_1, i + 1, j + 1, str_2),blur)
                ret, thresh = cv2.threshold(blur, threshold, 255 , 128, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite('hands_binarized_marcel_dataset/{}_{}_{}_binarized{}'.format(str_1, i + 1, j + 1, str_2),thresh)
                print('feito !')

binarize_set_hand_image_file('hands_marcel_dataset',1)