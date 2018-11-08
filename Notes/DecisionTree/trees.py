from math import log
import operator

def calcShannonEnt(dataSet):
    '''
    Func : Calculate the Shannon Entropy for a given dataSet 
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    #create dict for every possible category
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    #probability, and log(prob), sum 
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'Y'],
            [1,1,'Y'],
            [1,0,'N'],
            [0,1,'N'],
            [0,1,'N']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    '''
    Func: divide the dataset based on given feature
    '''
    # create new list object 
    retDataSet = []
    for featVec in dataSet:
        # Extraction
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #extract the data from dataset where the given feature has given value
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    choose the best feature to split
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0 ; bestFeature = -1
    #create unique feature list
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)# change list to set, keep unique keys.
        newEntropy = 0.0
        #calculate Entropy for every split way.
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #get bestInfoGain
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote]=0
        classCount[vote] += 11
    sortedClassCount = sorted(classCount.items(),\
     key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历玩所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #得到列表包含的所有属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
                        (dataSet,bestFeat,value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
    	if testVec[featIndex] == key:
    		if type(secondDict[key]).__name__ == 'dict':
    			classLabel = classify(secondDict[key],featLabels,testVec)
    		else: 	classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
	import pickle
	#fw = open(filename, 'wb')
	#pickle.dump(inputTree,fw)
	#fw.close()
	with open(filename, 'wb') as fw:
		pickle.dump(inputTree,fw)



def grabTree(filename):
	import pickle
	fr = open(filename,'rb')
	return pickle.load(fr)

def main():
    pass
    
if __name__ =="__main__":
    main()