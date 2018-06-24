import numpy as np
import cv2
import pandas as pd

def save_vectorized_load_faces_in_csv_file(vectors):
    #dataset_faces = vectorize_data_faces('samples_faces_dataset')
    #X, y = [dataset_faces[0], dataset_faces[1]]
    X, y = [vectors[0],vectors[1]]
    
    X_df = pd.DataFrame(X)
    X_df.to_csv('X.csv', index=False)

    y_df = pd.DataFrame(y)  
    y_df.to_csv('y.csv', index=False)

    print("Imagens vetorizadas e salvas em arquivos do tipo .csv")

def return_of_images_and_rotules_vectors():
    X = pd.read_csv('X.csv').values.tolist()
    #y = pd.read_csv('y.csv').values.tolist()
    y = np.asarray(pd.read_csv('y.csv').values.tolist()).transpose()[0] #vetorizando a lista y

    return [X,y]

def return_of_images_aligned_and_rotules_vectors():
    X = pd.read_csv('X_aligned.csv').values.tolist()
    #y = pd.read_csv('y.csv').values.tolist()
    y = np.asarray(pd.read_csv('y.csv').values.tolist()).transpose()[0] #vetorizando a lista y

    return [X,y]

def return_last_index_subject():
    last_index_subject = pd.read_csv('y.csv').values.tolist()[-1][0]

    return last_index_subject

def return_all_subject_names():

    return np.asarray(pd.read_csv('subjects_name.csv').values.tolist()).transpose()[0] 

def return_subject_name(subject_number):

    return np.asarray(pd.read_csv('subjects_name.csv').values.tolist())[subject_number - 1][0]