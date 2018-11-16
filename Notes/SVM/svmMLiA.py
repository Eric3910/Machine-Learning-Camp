import numpy as np 
import 


def loadDataSet(fileName):
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append(float(lineArr[0]),float(lineArr[1]))
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat

def selectJrand(i,m):
	j = i
	while(j == i):
		j = int(np.random.uniform(0,m))
	return j