from sklearn.model_selection import train_test_split
from vectorizeFaces import vectorize_data_faces,load_picture_captured
from saveReturnValuesCSV import return_of_images_and_rotules_vectors,return_of_images_aligned_and_rotules_vectors,return_subject_name
import numpy as np
from sklearn.preprocessing import normalize
from imageCapture import sample_capture_to_rank
from sklearn.neural_network import MLPClassifier
import pandas as pd

#inicializations
dataset_faces = return_of_images_aligned_and_rotules_vectors()
dataset_faces = return_of_images_and_rotules_vectors()
X, y = [dataset_faces[0],dataset_faces[1]]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

mlp = MLPClassifier()

def mpl_training():
	print("\n>>>>> Verificar precisão - accuracy - Classificador Mult-Layer Perceptron [MLP] <<<<<")


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	mlp.fit(X_train, y_train)

	predictions = mlp.predict(X_test)

	from sklearn.metrics import classification_report,confusion_matrix
	print("\n>> Informações Gerais <<\n")
	print(classification_report(y_test,predictions))
	print("\n>> Matriz de Confusão <<\n")
	print(pd.crosstab(y_test, mlp.predict(X_test),rownames=['Real'],colnames=['Predito'],margins=True))
	print("\n>> Média de acertos de precisões <<")
	print(np.mean(y_test==predictions))
	
def mlp_rank_a_sample(voice=False):
	
	#take a picture for classification
	#sample_capture_to_rank()
	
	#read the image that had captured
	img = load_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Padronizar imagem capturada 
	scaler.fit(img)
	img = scaler.transform(img)

	#Aplicar PCA - Testando ...
	#img = pca.fit_transform(img)

	mlp.fit(X,y)
	predict = mlp.predict(img)
	
	
	print("\nO Sistema prediz que você é : {}-{}\n".format(predict,return_subject_name(predict)))
	
