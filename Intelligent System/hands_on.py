import numpy as np
import cv2
from saveReturnValuesCSV import save_vectorized_load_faces_in_csv_file,return_last_index_subject
import subprocess
from vectorizeFaces import vectorize_data_faces_cutting,vectorize_data_faces
from detectAndAlignFaces import align_rank_sample
import imutils
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

#inicializations
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Função para capturar apenas uma imagem, para que o programa a classifique
def sample_capture_to_rank():
	#subprocess.call(["say","Para reconhecer você, será necessário que você se posicione em frente a camêra"])
	#subprocess.call(["say","Uma janela será aberta, e, quando estiver preparado aperte a tecla espaço para continuar!"])

	cam=cv2.VideoCapture(0)
	cv2.namedWindow("image_capture",cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image_capture', 600,600)

	print(">>> Pressione a tecla SPACE para capturar a imagem ou ESC para sair <<<\n")
	while True:
		ret, frame = cam.read()
		
		img_copy = frame.copy()

		#gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

		faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);          

		#Draw a rectangle around the faces
		for (x, y, w, h) in faces:
		    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		cv2.imshow('image_capture', frame)
		
		if not ret:
			break
		k = cv2.waitKey(1)

		if k%256 == 27:
			#ESC pressed
			print("ESC apertado, fechando a janela ...")
			break
		elif k%256 == 32:
			#SPACE pressed
			img_name = 'sample_to_rank.png'
			try:
				cv2.imwrite(img_name,cv2.cvtColor(align_a_sample(img_copy), cv2.COLOR_BGR2GRAY))
				print('Captura realizada com sucesso !\n')
				break 
			except:
				print('A captura não foi possível, por favor se posicionar adequadamente em frente a câmera e apertar a tecla ESPAÇO quando estiver preparado ...\n')

	cam.release()

	cv2.destroyAllWindows()

#sample_capture_to_rank()