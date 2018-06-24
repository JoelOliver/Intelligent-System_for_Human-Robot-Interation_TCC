import pandas as pd
import numpy as np
import cv2

import subprocess

def welcome_by_system(voice=False):
	if voice:
		subprocess.call(["say","Olá, seja bem vindo ao sistema de reconhecimento facial do Centauro"])
		subprocess.call(["say","Minha missão, pelo menos por enquanto, é tentar reconhecer você"])

welcome_by_system(voice=False)

from nearestNeighborsClassifier import knearest_neighborhood_training,centroid_training
from neuralNetworkClassifier import mpl_training
#Verificar taxas de acertos, treinamento e teste dos algorítmos
knearest_neighborhood_training()
centroid_training()
mpl_training()


from nearestNeighborsClassifier import knearest_rank_a_sample,nearest_centroid_rank_a_sample
from neuralNetworkClassifier import mlp_rank_a_sample

# Classificar uma amostra utilizando o classificador vizinho mais próximo
#knearest_rank_a_sample()

# Classificar uma amostra utilizando o classificador centroid mais próximo
#nearest_centroid_rank_a_sample()

# Classificar uma amostra utilizando o classificador neural MLP
#mlp_rank_a_sample()


from imageCapture import sample_capture_to_rank,samples_capture_to_dataBase

# Função para salvar novas imagens no Banco de Dados
#samples_capture_to_dataBase(8,20)


from vectorizeFaces import vectorize_data_faces,vectorize_data_faces_aligned
from saveReturnValuesCSV import save_vectorized_load_faces_in_csv_file

#save_vectorized_load_faces_in_csv_file(vectorize_data_faces('samples_faces_detected_dataset',7))	

from align_faces import align_faces
#align_faces('backup_samples_faces_dataset',7)