import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import imutils
import dlib
from imutils import face_utils

# Inicializations
D = 300 # Dimension to reshape images, when vectorize one vector -> len = 90000 (300x300)

def load_picture_captured():
    img_file = cv2.imread('sample_to_rank.png', 0)
    img = np.reshape(img_file, (img_file.shape[0] * img_file.shape[1]))
    return img

def vectorize_data_faces(Filename,numberOfPersons):
    # Number of Persons in DataBase
    N = numberOfPersons 

    # Number of image faces for each Person in Database
    Ni = 10

    # Data of images in vector format
    X = []

    # Target of each image (rotules)
    y = []

    # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = ['Subject_','subject0','subject']
    str_3 = ['.pgm','.png','.jpg']
    str_4 = '_type_'

    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes

            img_file = cv2.imread('{}/{}{}{}{}{}'.format(filename_str, str_1[0], i + 1, str_4, j + 1, str_3[1]),0)
            print('{}/{}{}{}{}{}'.format(filename_str, str_1[0], i + 1, str_4, j + 1, str_3[1]))
            #print(np.size(img_file))
            img = np.reshape(img_file, (img_file.shape[0]*img_file.shape[1]))
            X.append(img)
            y.append(i + 1)
            
    return [X, y]
