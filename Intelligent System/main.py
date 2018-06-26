import time

'''
## Para verificar o tempo de execução de qualquer função
inicio = time.time()
print(inicio)

some_function() #substituir aqui pela função verdadeira

fim = time.time()
print(fim)

print(fim - inicio)
'''

#from nearestNeighborsClassifier import knearest_neighborhood_training,centroid_training
#from neuralNetworkClassifier import mpl_training
#Verificar taxas de acertos, treinamento e teste dos algorítmos
#knearest_neighborhood_training()
#centroid_training()
#mpl_training()


#from nearestNeighborsClassifier import knearest_rank_a_sample,nearest_centroid_rank_a_sample
from neuralNetworkClassifier import mlp_rank_a_sample

# Classificar uma amostra utilizando o classificador vizinho mais próximo
#knearest_rank_a_sample()

# Classificar uma amostra utilizando o classificador centroid mais próximo
#nearest_centroid_rank_a_sample()

# Classificar uma amostra utilizando o classificador neural MLP
mlp_rank_a_sample()


#from imageCapture import sample_capture_to_rank,samples_capture_to_dataBase

# Função para salvar novas imagens no Banco de Dados
#samples_capture_to_dataBase(8,20)


#from vectorizeFaces import vectorize_data_faces,vectorize_data_faces_aligned
#from saveReturnValuesCSV import save_vectorized_load_faces_in_csv_file

#save_vectorized_load_faces_in_csv_file(vectorize_data_faces_aligned('samples_faces_aligned_dataset',7))	

#from align_faces import align_faces
#align_faces('backup_samples_faces_dataset',7)