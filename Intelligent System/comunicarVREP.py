"""
@author: joelribeiro
"""
import vrep
import sys
import time
from nearestNeighborsClassifier import knearest_rank_a_sample,nearest_centroid_rank_a_sample
from nearestNeighborsClassifier import knearest_rank_a_hand_sample,nearest_centroid_rank_a_hand_sample

print("\n<<<<<<<<<<< Bem vindo ao sistema inteligente para interação humano-robô >>>>>>>>>>>>>>>\n")
print(" Conexão sendo estabelecida com o simulador V-REP ... ")


vrep.simxFinish(-1) # apenas para o caso de haver outras conexões
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

#Declaracoes dos motores do robo Pioneer
errorCode, left_motor_handle = vrep.simxGetObjectHandle(clientID,"Pioneer_p3dx_leftMotor",vrep.simx_opmode_oneshot_wait)
errorCode, right_motor_handle = vrep.simxGetObjectHandle(clientID,"Pioneer_p3dx_rightMotor",vrep.simx_opmode_oneshot_wait)
velocityMotors = 0.5
stopMotors = 0.0


def controlPioneer(value):
	# Os resultados podem ser diferentes, dependendo das configurações da cena. Idealmente, considerar 
	# a simulação em tempo real
	print("O tipo de gesto foi : {}".format(value))
	# Fazer com que o robo pioneer se desloque para frente com cerca de 0.5 de velocidade, por 6 segundos
	if(value == 1):
	    vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,velocityMotors,vrep.simx_opmode_oneshot)
	    vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,velocityMotors,vrep.simx_opmode_oneshot)
	    time.sleep(6)
	    vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,stopMotors,vrep.simx_opmode_streaming)
	    vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,stopMotors,vrep.simx_opmode_streaming)
	# Fazer com que o robo pioneer se desloque 180 graus no sentido horário
	elif(value == 2):
	    vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,velocityMotors,vrep.simx_opmode_oneshot)
	    time.sleep(10)
	    vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,stopMotors,vrep.simx_opmode_streaming)
	# Fazer com que o robo pioneer se desloque 90 graus no sentido horário
	elif(value == 3):
	    vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,velocityMotors,vrep.simx_opmode_oneshot)
	    time.sleep(6)
	    vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,stopMotors,vrep.simx_opmode_streaming)
	# Fazer com que o robo pioneer se desloque 90 graus no sentido anti-horário
	elif(value == 4):
	    vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,velocityMotors,vrep.simx_opmode_oneshot)
	    time.sleep(6)
	    vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,stopMotors,vrep.simx_opmode_streaming)

if clientID!=-1:
    print ("A conexão foi estabelecida\n")
    
    face_autenticate = nearest_centroid_rank_a_sample()
    
    if(face_autenticate[0] == 2 or face_autenticate[0] == 3):
    	print("Olá {}".format(face_autenticate[1]))

    	while clientID!=-1:
	    	controlPioneer(knearest_rank_a_hand_sample())
    else:
    	print("\n!!! Usuário não reconhecido ou não autorizado !!!\n")

    # Stop simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
else:
    print ("Falha ao se comunicar com o VREP")
print ('Sistema finalizado')