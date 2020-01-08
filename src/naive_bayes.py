import json
from collections import Counter, defaultdict
import math
import random
import matplotlib.pyplot as plt
import seaborn as sn
from utils import getStemmedDocuments
import re
import nltk
from sklearn.metrics import confusion_matrix, f1_score
alpha = re.compile(r'[^A-Za-z]+')
import sys
import warnings
warnings.filterwarnings('ignore')

trainFile = sys.argv[1]
testFile = sys.argv[2]
partNum = sys.argv[3]

def accuracy(actual, predicted):
	if(len(actual) != len(predicted)):
		return 0
	count =0
	for a,p in zip(actual, predicted) :
		if(a==p):
			count += 1
	return count

def bigrams(text):
	nltk_tokens = nltk.word_tokenize(text)
	return list(nltk.bigrams(nltk_tokens))
	
def readJson(filename):
	text =[]
	stars =[]
	for line in open(filename, mode="r"):
		x = json.loads(line)
		stars.append(x["stars"])
		text.append(x["text"].strip())
	return text, stars


textTrain, starsTrain = readJson(trainFile)
textTest , starsTest = readJson(testFile)
stars = list(sorted(set(starsTest)))

def classify(text ,priors, wordCountClass , totalWordsClass , totalWordsDict, feature ):
	probs =priors.copy()
	for cls in priors.keys():
		if feature == "bigrams":
			featureArray = bigrams(text)
		else:
			featureArray = text.split()
		for word in featureArray:
			# Count of a word may be zero for two reasons:
			# 1 - Word did not occur in reviews of that class
			# 2 - Word did not occur in the entire vocabulary
			count = wordCountClass[cls].get(word,0)
			
			# caclculate the probablity to be classified in this class using laplace smoothing where c =1
			probs[cls] += math.log((count +1) / (totalWordsClass[cls] + totalWordsDict) )
	return max(probs, key= probs.get)
	

def train(textTrain, textTest, feature):
	# Store counts of each word in documents of each class
	wordCountClass = defaultdict(Counter)

	# Store the number of documents that contain this word
	wordCountDocument = Counter()

	# Total words in documents of a class
	totalWordsClass ={}

	# priors of all class
	priors = {}

	def updateWordCount(t):
		text = t[0]
		#print(text)
		cls = t[1]
		if feature == "bigrams":
			words = bigrams(text)
		else:
			words = text.split()
		wordCountDocument.update(set(words))
		wordCountClass[cls].update(words)
		return
	countClass = Counter(starsTrain)
	totalDocuments = len(textTrain)
	for cls in countClass.keys():
		priors[cls] =  countClass[cls] / totalDocuments
	x = list(map(updateWordCount ,zip(textTrain, starsTrain)))
	
	# tf-idf word count 
	
	if feature == "tf-idf" :
		for cls , ctr in wordCountClass.items():
			for word in ctr:
				ctr[word] = ctr[word] * math.log(totalDocuments / wordCountDocument[word])
			totalWordsClass[cls] = sum(ctr.values())
	else:

		# normal count
		for cls, ctr in wordCountClass.items():
			totalWordsClass[cls] = sum(ctr.values())
	
	
	totalWordsDict = len(wordCountDocument)
	
	return priors, wordCountClass, totalWordsClass, totalWordsDict 



def naiveBayes_a(trainData = textTrain, testData = textTest, feature = None, accuracyTrain = False):
	priors, wordCountClass , totalWordsClass , totalWordsDict = train(trainData, testData, feature)
	if accuracyTrain:
		predictedTrain = []
		for text in trainData:
			predictedTrain.append(classify(text, priors, wordCountClass , totalWordsClass , totalWordsDict, feature))
		correct = accuracy(starsTrain, predictedTrain)
		print("Training accuracy: ", (correct * 100) / len(starsTrain)) 
	predictedTest = []
	for text in testData:
		predictedTest.append(classify(text, priors, wordCountClass , totalWordsClass , totalWordsDict, feature))
	correct = accuracy(starsTest, predictedTest)
	acc = (correct * 100) / len(starsTest)
	print("Testing accuracy: ", acc) 
	return predictedTest

def naiveBayes_b():
	classCount = Counter(starsTrain)
	ratings = list(sorted(classCount.keys()))
	
	# Randomly classified
	randomPrediction = [random.choice(ratings) for _ in range(len(textTest))]
	randomAccuracy = accuracy(starsTest, randomPrediction)
	
	print("Random Testing Accuracy : " , (randomAccuracy * 100) / len(textTest) )
	stars = list(sorted(set(starsTest)))
	print("F-score: ")
	f_score = f1_score(starsTest, randomPrediction, average=None, labels = stars)
	print(f_score)
	f_score_avg = f1_score(starsTest, randomPrediction, labels = stars, average='macro')
	print("Macro -f -Score :", f_score_avg)
	
	# Majority Classification
	
	majorityClass = max(classCount, key = classCount.get)
	majorityPrediction = [majorityClass for _ in range(len(textTest))]
	majorityAccuracy = accuracy(starsTest, majorityPrediction)
	
	print("Majority Testing Accuracy : " , (majorityAccuracy * 100) / len(textTest) )
	
	print("F-score: ")
	f_score = f1_score(starsTest, majorityPrediction, average=None, labels = stars)
	print(f_score)
	f_score_avg = f1_score(starsTest, majorityPrediction, labels = stars, average='macro')
	print("Macro -f -Score :", f_score_avg)



#Plot Confusion Matrix for testing and Training Accuracies
def naiveBayes_c(testingClassification):
	ratings = list(sorted(set(starsTest)))
	confusionMatrix = {}
	for cls in ratings:
		confusionMatrix[cls] = {cc: 0 for cc in ratings}
	for (actual ,predicted) in zip(starsTest , testingClassification) :
		confusionMatrix[actual][predicted] += 1
	arrayMatrix = []
	for cls in sorted(confusionMatrix.keys()):
		tempArr = []
		for c in sorted(confusionMatrix[cls].keys()):
			tempArr.append(confusionMatrix[cls][c])
		arrayMatrix.append(tempArr)
	acc = " - %.2f%% accuracy" % (accuracy(starsTest, testingClassification) * 100 / len(starsTest))


	plt.figure(figsize=(10, 7))

	ax = sn.heatmap(arrayMatrix, fmt="d", annot=True, cbar=False,
                    cmap=sn.cubehelix_palette(15),
                    xticklabels=ratings, yticklabels=ratings)
	# Move X-Axis to top
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	
	ax.set(xlabel="Predicted", ylabel="Actual")

	plt.title("Confusion Matrix for Testing data - " + acc , y = 1.08 , loc = "center")
	plt.savefig("ConfusionMatrix.png")
	plt.show()


def naiveBayes_d_e_f(partNum):
	textStemmedTrain =[]
	for i in textTrain :
		#text = re.sub(alpha, ' ', i).strip()
		tokens = getStemmedDocuments(i, False)
		textStemmedTrain.append(tokens)
	
	textStemmedTest =[]
	for i in textTest :
		#text = re.sub(alpha, ' ', i).strip()
		tokens = getStemmedDocuments(i , False)
		textStemmedTest.append(tokens)
	if(partNum == "d"):
		stars = list(sorted(set(starsTest)))
		print("\n Stemmed Data Training : \n")
		prediction = naiveBayes_a(textStemmedTrain, textStemmedTest, feature = None, accuracyTrain = True)
		print("F-score: ")
		f_score = f1_score(starsTest, prediction, average=None, labels = stars)
		print(f_score)
		f_score_avg = f1_score(starsTest, prediction, labels = stars, average='macro')
		print("Macro -f -Score :", f_score_avg)
		
	else:
		stars = list(sorted(set(starsTest)))
		# Part_e
		print("\n Bigrams Data Training : \n")
		prediction = naiveBayes_a(textStemmedTrain, textStemmedTest, feature = "bigrams", accuracyTrain = False)
		print("F-score: ")
		f_score = f1_score(starsTest, prediction, average=None, labels = stars)
		print(f_score)

		f_score_avg = f1_score(starsTest, prediction, labels = stars, average='macro')
		print("Macro -f -Score :", f_score_avg)

		print("\n TF-IDF Data Training : \n")
		prediction = naiveBayes_a(textStemmedTrain, textStemmedTest, feature = "tf-idf", accuracyTrain = False)

		stars = list(sorted(set(starsTest)))

		# part f
		print("F-score: ")
		f_score = f1_score(starsTest, prediction, average=None, labels = stars)
		print(f_score)

		f_score_avg = f1_score(starsTest, prediction, labels = stars, average='macro')
		print("Macro -f -Score :", f_score_avg)


def naiveBayes_g():
	'''textStemmedTrain =[]
	for i in textTrain :
		#text = re.sub(alpha, ' ', i).strip()
		tokens = getStemmedDocuments(i, False)
		textStemmedTrain.append(tokens)
	
	textStemmedTest =[]
	for i in textTest :
		#text = re.sub(alpha, ' ', i).strip()
		tokens = getStemmedDocuments(i , False)
		textStemmedTest.append(tokens)'''
	print("\n Bigrams Data Training : \n")
	prediction = naiveBayes_a(feature = "bigrams", accuracyTrain = False)
	print("F-score: ")
	f_score = f1_score(starsTest, prediction, average=None, labels = stars)
	print(f_score)

	f_score_avg = f1_score(starsTest, prediction, labels = stars, average='macro')
	print(f_score_avg)


if(partNum == "a"):
	stars = list(sorted(set(starsTest)))
	testingClassification = naiveBayes_a(feature = None, accuracyTrain = True)
	print("F-score: ")
	f_score = f1_score(starsTest, testingClassification, average=None, labels = stars)
	print(f_score)
	f_score_avg = f1_score(starsTest, testingClassification, labels = stars, average='macro')
	print("Macro -f -Score :", f_score_avg)

elif(partNum == "b"):
	naiveBayes_b()
elif(partNum == "c"):
	stars = list(sorted(set(starsTest)))
	testingClassification = naiveBayes_a(feature = None, accuracyTrain = True)
	naiveBayes_c(testingClassification)
	print("F-score: ")
	f_score = f1_score(starsTest, testingClassification, average=None, labels = stars)
	print(f_score)
	f_score_avg = f1_score(starsTest, testingClassification, labels = stars, average='macro')
	print("Macro -f -Score :", f_score_avg)
elif(partNum == "d" or partNum == "e"):
	naiveBayes_d_e_f(partNum)
elif(partNum == "g"):
	naiveBayes_g()
else:
	print("wrong Part number")
	#naiveBayes_a(feature = "bigrams", accuracyTrain = False)
	
