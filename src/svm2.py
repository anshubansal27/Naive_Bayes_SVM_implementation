from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from numpy import linalg as la
from collections import Counter

import numpy as np
from svmutil import *
import sys
import time

c = 1.0
gamma = 0.05

trainFile = sys.argv[1]
testFile = sys.argv[2]
partNum = sys.argv[3]

# reading data from file for label 2 and 3 and preprocessing of data
def readData(filename, delimeter = ","):
	x = np.loadtxt(filename, delimiter = delimeter)
	m,n = x.shape
	# filter the data
	x = x[ (x[:,n-1] == 2) | (x[:,n-1] == 3) ]
	m, n = x.shape
	label = np.zeros((m,1))
	for i in range(m):
		for j in range(n-1):
			if not x[i][j] == 0:
				x[i][j] = x[i][j]/ 255.0
		label[i] = 1 if x[i][n-1] == 2 else -1
	x = np.delete(x, n-1, 1)
	m, n = x.shape
	return x, label, m, n
	


trainData,labelTrain, mtrain, ntrain = readData(trainFile)
testData, labelTest, mtest , ntest = readData(testFile)

kMatrix = np.zeros((mtrain, mtrain))

def linearKernelMatrix():
	for i in range(mtrain):
		for j in range(mtrain):
			kMatrix[i][j] = np.dot(trainData[i], trainData[j])

def gaussianKernelKmatrix():
	for i in range(mtrain):
		for j in range(mtrain):
			temp = la.norm((trainData[i], trainData[j])) ** 2
			kMatrix[i][j] = np.exp(-temp * gamma)


if partNum == "a" or partNum =="b" :
	start_time = time.time()
	if partNum == "a":
		linearKernelMatrix()
	elif partNum == "b":
		gaussianKernelKmatrix()
	P = cvxopt_matrix(np.outer(labelTrain,labelTrain) * kMatrix)
	q = cvxopt_matrix(-np.ones((mtrain, 1)))
	A = cvxopt_matrix(labelTrain.reshape(1, -1))
	b = cvxopt_matrix(np.zeros(1))
	h = cvxopt_matrix(np.hstack((np.zeros(mtrain), np.ones(mtrain)*c)))
	y = np.eye(mtrain)
	x = np.eye(mtrain) * -1
	G = cvxopt_matrix(np.vstack((x, y)))


	solution = cvxopt_solvers.qp(P, q, G, h, A, b)

	alpha = np.array(solution['x'])


	#Selecting the set of indices S corresponding to non zero parameters
	S = (alpha > 1e-5).flatten()

	if partNum == "a" :
		#w parameter in vectorized form
		w = ((alpha * labelTrain).T @ trainData)

		print("parameter w", w.shape, trainData[0].shape)
		print(w)

		maximum = None
		minimum = None

		w = w.flatten()

		for i in range(mtrain):
			res = np.dot(w, trainData[i])
			if(labelTrain[i] == -1):
				if(maximum is None or maximum < res) :
					maximum = res
			else:
				if(minimum is None or minimum > res):
					minimum = res

		b = -(maximum + minimum ) /2

		print("b" , b)
		
		print("--cvxopt linear %s seconds ---" % (time.time() - start_time))

		correct = 0

		for i in range(mtest):
			result = np.dot(w, testData[i]) + b
			if((result < 0 and labelTest[i] == -1) or (result > 0 and labelTest[i] == 1)):
				correct +=1

		print("Correct test cases" , correct , (correct / mtest) * 100)
	else:
		index = np.array(range(len(alpha)))[S]
		alpha = alpha[S]
		SVx = trainData[S]
		SVy = labelTrain[S]
		b =0.0
		for n in range(len(SVx)):
			b += SVy[n]
			b -= np.sum(alpha * SVy * kMatrix[index[n],S])
		b /= len(alpha)
		print("Parameter b: ", b)
		print("---cvxopt gaussian %s seconds ---" % (time.time() - start_time))
		labelPredict = np.zeros(mtest)
		for i in range(mtest):
			s = 0
			for a, sv_y, sv in zip(alpha, SVy, SVx):
				temp = np.exp(- (np.linalg.norm(testData[i] -sv )**2 ) * gamma)
				s += a * sv_y * temp
			labelPredict[i] = s
		labelPredict = labelPredict + b
		
		correct =0
		for i in range(mtest):
			if((labelPredict[i] < 0 and labelTest[i] == -1) or (labelPredict[i] > 0 and labelTest[i] == 1)):
				correct +=1
		print("Correct test cases" , correct , (correct / mtest) * 100)
		
		

elif partNum == "c":
	#LibSVM
	def converttoList(a):
		return a.flatten()

	start_time = time.time()
	parameter = svm_parameter('-t 0')
	problem = svm_problem(converttoList(np.asarray(labelTrain)), np.asarray(trainData).tolist())
	model = svm_train(problem, parameter)
	print("---LIBSVM Linear %s seconds ---" % (time.time() - start_time))
	_, p_acc, _ = svm_predict(converttoList(np.asarray(labelTest)), np.asarray(testData).tolist(), model)
	print("Accuracy with linear kernel", p_acc[0])
	print("b value : " , model.rho[0])

	parameter = svm_parameter('-q -t 2 -g 0.05')
	problem = svm_problem(converttoList(np.asarray(labelTrain)), np.asarray(trainData).tolist())
	model = svm_train(problem, parameter)
	print("---LIBSVM Gaussian %s seconds ---" % (time.time() - start_time))
	_, p_acc, _ = svm_predict(converttoList(np.asarray(labelTest)), np.asarray(testData).tolist(), model)
	print("Accuracy with Gaussian kernel" , p_acc[0])
	print("b value : " , model.rho[0])

else:
	print("Wrong part number")










