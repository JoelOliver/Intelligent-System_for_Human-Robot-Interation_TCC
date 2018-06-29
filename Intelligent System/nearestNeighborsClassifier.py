from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from vectorizeFaces import vectorize_data_faces,load_picture_captured
from saveReturnValuesCSV import return_of_images_and_rotules_vectors,return_of_images_hands_and_rotules_vectors,return_subject_name
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from facesCapture import sample_capture_to_rank
from handsCapture import sample_hand_capture_to_rank
import pandas as pd
from vectorizeHands import load_hand_picture_captured


#inicializations for image faces
dataset_faces = return_of_images_and_rotules_vectors()
X, y = [dataset_faces[0],dataset_faces[1]]

#inicializations for image hands
dataset_faces = return_of_images_hands_and_rotules_vectors()
X_hand, y_hand = [dataset_faces[0],dataset_faces[1]]


#Normalization of X Matrix of image_vectors
#X = normalize(X)

from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)

#PCA - Test
from sklearn.decomposition import PCA, IncrementalPCA
#n_components = 9000
#ipca = IncrementalPCA(n_components=n_components)
#X = ipca.fit_transform(X)

#pca = PCA(n_components=n_components)
#X = pca.fit_transform(X)


# para mudar valores dos parâmetros, verificar documentação do scikit-learn
neighKNeigh = KNeighborsClassifier(n_neighbors=3)
neighCentroid = NearestCentroid()

def knearest_neighborhood_training():
	print("\n>>>>> Verificar precisão - accuracy - Classificador KNearest Neighborhood <<<<<")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	
	neighKNeigh.fit(X_train, y_train)

	predictions = neighKNeigh.predict(X_test)

	from sklearn.metrics import classification_report,confusion_matrix
	print("\n>> Informações Gerais <<\n")
	print(classification_report(y_test,predictions))
	print("\n>> Matriz de Confusão <<\n")
	print(pd.crosstab(y_test, neighKNeigh.predict(X_test),rownames=['Real'],colnames=['Predito'],margins=True))
	print("\n>> Média de acertos de precisões <<\n")
	print(np.mean(y_test==predictions))


#Testar função -> nearest_knearest_neighborhood_training() 
#knearest_neighborhood_training() 

def centroid_training():
	print("\n>>>>> Verificar precisão - accuracy - Classificador Nearest Centroid <<<<<")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	
	neighCentroid.fit(X_train, y_train)

	predictions = neighCentroid.predict(X_test)

	from sklearn.metrics import classification_report,confusion_matrix
	print("\n>> Informações Gerais <<\n")
	print(classification_report(y_test,predictions))
	print("\n>> Matriz de Confusão <<\n")
	print(pd.crosstab(y_test, neighCentroid.predict(X_test),rownames=['Real'],colnames=['Predito'],margins=True))
	print("\n>> Média de acertos de precisões <<\n")
	print(np.mean(y_test==predictions))


def knearest_rank_a_hand_sample():

	#take a picture for classification

	sample_hand_capture_to_rank()

		
	#read the image that had captured
	img = load_hand_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Normalizar imagem capturada e o dataset
	#img = normalize(img)
	
	neighKNeigh.fit(X_hand,y_hand)
	predict = neighKNeigh.predict(img)
	

	return predict[0]

def nearest_centroid_rank_a_hand_sample():

	#take a picture for classification

	sample_hand_capture_to_rank()

		
	#read the image that had captured
	img = load_hand_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Normalizar imagem capturada e o dataset
	#img = normalize(img)
	
	neighCentroid.fit(X_hand,y_hand)
	predict = neighKNeigh.predict(img)
	

	return predict[0]

def knearest_rank_a_sample():
	
	#take a picture for classification

	sample_capture_to_rank()

	
	#read the image that had captured
	img = load_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Normalizar imagem capturada e o dataset
	#img = normalize(img)

	#Aplicar PCA - Testando ...
	#img = pca.fit_transform(img)

	neighKNeigh.fit(X,y)
	predict = neighKNeigh.predict(img)


	print("\nO Sistema prediz que você é : {}-{}\n".format(predict,return_subject_name(predict)))
	return [predict[0], return_subject_name(predict)]

def nearest_centroid_rank_a_sample():
	
	#take a picture for classification

	sample_capture_to_rank()

	
	#read the image that had captured
	img = load_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Normalizar imagem capturada e o dataset
	#img = normalize(img)

	#Aplicar PCA - Testando ...
	#img = pca.fit_transform(img)

	neighCentroid.fit(X,y)
	predict = neighCentroid.predict(img)


	print("\nO Sistema prediz que você é : {}-{}\n".format(predict,return_subject_name(predict)))
	return [predict[0], return_subject_name(predict)]
