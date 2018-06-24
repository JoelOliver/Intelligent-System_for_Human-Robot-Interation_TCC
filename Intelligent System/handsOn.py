import numpy as np
import cv2
from saveReturnValuesCSV import return_of_image_and_rotule_vectors

dataset_faces = return_of_image_and_rotule_vectors()
X, y = [dataset_faces[0],dataset_faces[1]]

print(len(X[0]))
