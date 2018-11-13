import numpy as np

def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inX):
	return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose() # 对label行向量转置
	m,n = np.shape(dataMatrix)
	alpha = 0.001 # 移动步长
	maxCycles = 500 # 迭代次数
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

def plotBestFit(weights):
	import matplotlib.pyplot as plt
	dataMat, labelMat = loadDataSet()
	dataArr = np.array(dataMat)
	n = np.shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1, s=30 ,c ='red', marker='s')
	ax.scatter(xcord2,ycord2, s=30 ,c ='green')
	x = arange(-3.0,3.0,0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.xlabel('X2')
	plt.show()

def stocGradAscent0(dataMatrix, classLabels):
	m,n = np.shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(n):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

def stocGradAscent1(dataMatrix, classLabels, numIter):
	m,n = np.shape(dataMatrix)
	weights = np.ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i)+0.01 # alpha 随着迭代次数逐渐减小，并且，j是迭代次数，i是样本点的下标。这样当j<<max(i)的时候，alpha不是严格下降。
			randIndex = int(np.random.uniform(0,len(dataIndex)))# 选择随机数
			h = sigmoid(sum(dataMatrix[randIndex]*weights)) 
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])#将该样本删除
	return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0



def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = [] ; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,500)
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr),trainWeights))!= int(currLine[21]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print("The error Rate of this test is : %f"%errorRate)
	return errorRate

def multiTest():
	numTests = 10; errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print("after %d iterations the average error rate is %f"%(numTests,errorSum/float(numTests)))
