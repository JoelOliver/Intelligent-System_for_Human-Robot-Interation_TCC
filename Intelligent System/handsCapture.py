import cv2
import numpy as np
import copy
import math
import time

def sample_hand_capture_to_rank():
# parameters
    cap_region_x_begin=0.5  # start point/total width
    cap_region_y_end=0.8  # start point/total width
    threshold = 60  #  BINARY threshold
    blurValue = 41  # GaussianBlur parameter
    bgSubThreshold = 50

    global learningRate, frame_cut, img_mask, blur, thresh
    learningRate = 0
    frame_cut = 0
    img_mask = 0
    blur = 0
    thresh = 0


    # variables
    global isBgCaptured, triggerSwitch
    isBgCaptured = 0   # bool, whether the background captured
    triggerSwitch = False  # if true, keyborad simulator works

    def printThreshold(thr):
        print("! Changed threshold to "+str(thr))


    def removeBG(frame):
        fgmask = bgModel.apply(frame,learningRate=learningRate)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    # Camera
    camera = cv2.VideoCapture(0)
    camera.set(10,200)
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
    cv2.namedWindow('original',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 600,600)

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)

        #  Main operation
        if isBgCaptured == 1:  # this part wont run until background captured
            frame_cut = frame[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]].copy()
            img_mask = removeBG(frame)
            img_mask = img_mask[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            #cv2.imshow('mask', img_mask)
            #cv2.imshow('test',frame[0:int(cap_region_y_end * frame.shape[0]),
            #            int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] .copy())

            #cv2.imwrite('test.png',frame[0:int(cap_region_y_end * frame.shape[0]),
            #            int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] .copy())


            # convert the image into binary image
            gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            #cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            #cv2.imshow('ori', thresh)
            
            k = ord('r')



        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            break
        elif k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print( '!!!Background Captured!!!')
        elif k == ord('r'):  # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0

            cv2.imwrite('sample_hand_to_rank.png',thresh.copy())
            break

#sample_hand_capture_to_rank()
def hands_capture_to_database():
# parameters
    cap_region_x_begin=0.5  # start point/total width
    cap_region_y_end=0.8  # start point/total width
    threshold = 60  #  BINARY threshold
    blurValue = 41  # GaussianBlur parameter
    bgSubThreshold = 50

    global learningRate, frame_cut, img_mask, blur, thresh
    learningRate = 0
    frame_cut = 0
    img_mask = 0
    blur = 0
    thresh = 0


    # variables
    global isBgCaptured, triggerSwitch
    isBgCaptured = 0   # bool, whether the background captured
    triggerSwitch = False  # if true, keyborad simulator works

    def printThreshold(thr):
        print("! Changed threshold to "+str(thr))


    def removeBG(frame):
        fgmask = bgModel.apply(frame,learningRate=learningRate)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    # Camera
    camera = cv2.VideoCapture(0)
    camera.set(10,200)
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
    cv2.namedWindow('original',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 600,600)

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)

        #  Main operation
        if isBgCaptured == 1:  # this part wont run until background captured
            frame_cut = frame[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]].copy()
            img_mask = removeBG(frame)
            img_mask = img_mask[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            #cv2.imshow('mask', img_mask)
            cv2.imshow('test',frame[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] .copy())

            #cv2.imwrite('test.png',frame[0:int(cap_region_y_end * frame.shape[0]),
            #            int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] .copy())


            # convert the image into binary image
            gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            #cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('ori', thresh)


        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            break
        elif k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print( '!!!Background Captured!!!')
        elif k == ord('r'):  # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            i=15
            cv2.imwrite('hand_4_{}.png'.format(i),frame_cut.copy())
            cv2.imwrite('hand_mask_4_{}.png'.format(i),img_mask.copy())
            cv2.imwrite('hand_blur_4_{}.png'.format(i),blur.copy())
            cv2.imwrite('hand_binarized_4_{}.png'.format(i),thresh.copy())
            break

#hands_capture_to_database()
