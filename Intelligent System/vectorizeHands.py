import numpy as np
import cv2
import pandas as pd


# Inicializations
D = 100 # Dimension to reshape images, when vectorize one vector -> len = (200x200)

def load_hand_picture_captured():
    img_file = cv2.imread('sample_hand_to_rank.png', 0)
    img = cv2.resize(img_file, (D, D))
    img = np.reshape(img, (D * D))
#    img = np.reshape(img_file, (img_file.shape[0] * img_file.shape[1]))
    return img


def vectorize_data_hands(Filename,numberOfHands):
    # Number of Hands Class in DataBase
    N = numberOfHands 

    # Number of image hands for each Sample in Database
    Ni = 15

    # Data of images in vector format
    X = []

    # Target of each image (rotules)
    y = []

    # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = 'hand'
    str_2 = '.png'

    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes
            
            img_file = cv2.imread('{}/{}_binarized_{}_{}{}'.format(filename_str, str_1, i + 1, j + 1, str_2),0)
            #print('{}'.format(img_file))
            #print(np.size(img_file))
            img = cv2.resize(img_file, (D, D))
            img = np.reshape(img,(D*D))
            #img = np.reshape(img_file, (img_file.shape[0] * img_file.shape[1]))

            X.append(img)
            y.append(i + 1)
            
    return [X, y]

#from saveReturnValuesCSV import save_vectorized_load_hands_in_csv_file
#save_vectorized_load_hands_in_csv_file(vectorize_data_hands('samples_hands_binarized_dataset',4))
