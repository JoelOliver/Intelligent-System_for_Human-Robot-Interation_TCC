import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import imutils
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

def align_faces(Filename,numberOfPersons):
    
    # Number of Persons in DataBase
    N = numberOfPersons 

    # Number of image faces for each Person in Database
    Ni = 10

    # Data of images in vector format
    X = []

    # Target of each image (rotules)
    y = []

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = ['Subject_','subject0','subject']
    str_3 = ['.pgm','.png','.jpg']
    str_4 = '_type_'

    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes
            #img_file = cv2.imread('{}/{}{}{}{}{}'.format(filename_str, str_1[0], i + 1, str_4, j + 1, str_3[1]),0)
            #print('{}'.format(img_file))
            #print(np.size(img_file))
            img_file = cv2.imread('{}/{}{}{}{}{}'.format(filename_str, str_1[0], i + 1, str_4, j + 1, str_3[1]))
            img_file = imutils.resize(img_file, width=800)
            gray = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 2)

            # loop over the face detections
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                faceOrig = imutils.resize(img_file[y:y + h, x:x + w], width=256)
                faceAligned = fa.align(img_file, gray, rect)

                print('{}{}{}{}{} aligned'.format(str_1[0], i + 1, str_4, j + 1, str_3[1]))
                cv2.imwrite('saples_faces_aligned_dataset/{}{}{}{}{}'.format(str_1[0], i + 1, str_4, j + 1, str_3[1]), faceAligned)

def align_a_sample(img):

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    img = imutils.resize(img, width=800)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 2)

    faceAligned = 0
    
    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        #faceOrig = imutils.resize(img[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(img, gray, rect)

    return faceAligned
                       
    