from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from numpy import linalg as la
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from svmutil import *
import sys
import math
import seaborn as sn
import time

c = 1.0
gamma = 0.05

trainFile = sys.argv[1]
testFile = sys.argv[2]
partNum = sys.argv[3]

def splitData(data, label, testSize = 0.10):
	m = len(data)
	splitindex = math.ceil(m *(1-testSize))
	splitdata = np.split(data, [splitindex])
	splitlabel = np.split(label, [splitindex])
	return splitdata, splitlabel

# reading data from file for label 2 and 3 and preprocessing of data
def readData(filename, delimeter = ","):
	x = np.loadtxt(filename, delimiter = delimeter)
	m,n = x.shape
	label = np.zeros((m,1))
	for i in range(m):
		for j in range(n-1):
			if not x[i][j] == 0:
				x[i][j] = x[i][j]/ 255.0
		label[i] = x[i][n-1]
	x = np.delete(x, n-1, 1)
	m, n = x.shape
	return x, label, m, n

trainData,labelTrain, mtrain, ntrain = readData(trainFile)
testData, labelTest, mtest , ntest = readData(testFile)


#Plot Confusion Matrix for testing and Training Accuracies
def svmConfusionMatrix(actuallabels, predictedlabels, acc, fname):
	ratings = list(sorted(set(actuallabels)))
	confusionMatrix = {}
	for cls in ratings:
		confusionMatrix[cls] = {cc: 0 for cc in ratings}
	for (actual ,predicted) in zip(actuallabels , predictedlabels) :
		confusionMatrix[actual][predicted] += 1
	arrayMatrix = []
	for cls in sorted(confusionMatrix.keys()):
		tempArr = []
		for c in sorted(confusionMatrix[cls].keys()):
			tempArr.append(confusionMatrix[cls][c])
		arrayMatrix.append(tempArr)
	#acc = " - %.2f%% accuracy" % (accuracy(actuallabels, testingClassification) * 100 / len(actuallabels))


	plt.figure(figsize=(10, 7))

	ax = sn.heatmap(arrayMatrix, fmt="d", annot=True, cbar=False,
                    cmap=sn.cubehelix_palette(15),
                    xticklabels=ratings, yticklabels=ratings)
	# Move X-Axis to top
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	
	ax.set(xlabel="Predicted", ylabel="Actual")

	plt.title(fname + " with accuracy = " + str(acc) , y = 1.08 , loc = "center")
	plt.savefig(fname + ".jpg")
	plt.show()
	


def converttoList(a):
	return a.flatten()

if(partNum == "b" or partNum == "c"):
	start_time = time.time()
	parameter = svm_parameter('-q -t 2 -g 0.05')
	problem = svm_problem(converttoList(np.asarray(labelTrain)), np.asarray(trainData).tolist())
	model = svm_train(problem, parameter)
	print("---LIBSVM Gaussian %s seconds ---" % (time.time() - start_time))
	trainPredicted, p_acc, _ = svm_predict(converttoList(np.asarray(labelTrain)), np.asarray(trainData).tolist(), model)
	print("Accuracy with Gaussian kernel on Training" , p_acc[0])
	if(partNum == "c"):
		svmConfusionMatrix(converttoList(labelTrain), trainPredicted, p_acc[0], "Confusion Matrix training")
	testPredicted, p_acc, _ = svm_predict(converttoList(np.asarray(labelTest)), np.asarray(testData).tolist(), model)
	print("Accuracy with Gaussian kernel on Testing" , p_acc[0])
	if(partNum == "c"):
		svmConfusionMatrix(converttoList(labelTest), testPredicted, p_acc[0], "Confusion Matrix testing")

elif(partNum == "d"):

	data , label = splitData(trainData, labelTrain, testSize = 0.10)
	traindata = data[0]
	testdata = data[1]
	labeltrain = label[0]
	labeltest = label[1]

	validation_acc =[]
	test_acc = []
	carr = [0.00001, 0.001, 1.0, 5.0, 10.0]
	problem = svm_problem(converttoList(np.asarray(labeltrain)), np.asarray(traindata).tolist())
	for c in carr :
		c = " -c %f " % c
		parameter = svm_parameter('-q -t 2 -g 0.05 ' + c)
		model = svm_train(problem, parameter)
		trainPredicted, p_acc, _ = svm_predict(converttoList(np.asarray(labeltest)), np.asarray(testdata).tolist(), model)
		validation_acc.append(p_acc[0])
		testPredicted, p_acc, _ = svm_predict(converttoList(np.asarray(labelTest)), np.asarray(testData).tolist(), model)
		test_acc.append(p_acc[0])

	print(validation_acc, test_acc)

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	plt.title("C versus Accuracy")
	line1, = ax.plot(carr,validation_acc , color='blue', lw=2, label = "validation accuracy" , marker = "o")
	line2, = ax.plot(carr,test_acc , color='red', lw=2, label = "testing accuracy", marker = "x")
	ax.set_xlabel("C")
	ax.set_ylabel("Accuracy")
	plt.legend(handles = [line1, line2])

	ax.set_xscale('log')
	plt.show()
else:
	print("Wrong Part Number")


 
