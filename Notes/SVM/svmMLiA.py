from numpy import *
from time import sleep

def loadDataSet(fileName):
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat

def selectJrand(i,m):
	j = i
	while(j == i):
		j = int(random.uniform(0,m))
	return j

def clipAlpha(aj,H,L):
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    Func:
    Paramaters:
    dataMatIn: input data matrix
    classLabels : class labels
    C: a fixed number
    toler: tolerance rate
    maxIter: max iternate times
    '''
    # 标签转置为列向量。类别标签向量元素的每一行对应数据矩阵中的每一行。
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose() 
    #m,n是数据矩阵的大小。
    b = 0; m,n = shape(dataMatrix)
    #构建alpha矩阵，初始化元素为0.
    alphas = mat(zeros((m,1)))
    #iter变量用于记录alpha没有任何改变的情况下遍历数据集的次数
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        #对整个集合顺序遍历，变量alphaPairsChanged用于记录alpha是否已经进行优化。
        for i in range(m):
            #fXi是预测的类别，np.multiply是矩阵乘法，.T是矩阵转置。
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # Ei是预估值与实际结果的误差。
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            #随机选择第二个alpha = j，如果相对误差率超过容错率绝对值并且保证alphas[i]在0到C之间。
            #因为如果达到0或者C，则说明已经在边界上，就不能再调整了。
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #selectJrand是用于随机选择一个j
                j = selectJrand(i,m)
                #再次计算在j下的分类结果
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                #计算j情况下实际值与预估值的误差
                Ej = fXj - float(labelMat[j])
                #旧的alpha和新的alpha通过copy()生成副本，如果简单赋值，只是传递了列表引用，后面的计算会更改，无法达到比较新旧矩阵的目的
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                #计算L和H，如果两个alpha对应的数据标签是相反的（标签只有两个结果）：
                #否则如果数据标签是相同的，则L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                #如果L=H，就直接跳出本次，进入下一次的for循环
                if L==H: print("L==H"); continue
                #eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                #基于eta对alphas[j]进行修改
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                #利用clipAlpha对alphas[j]进行调整
                alphas[j] = clipAlpha(alphas[j],H,L)
                #判断alphas[j]跟调整前是否有轻微改变，如果是就跳出循环。
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                #alphas[i]同样调整这么大，但是是反方向。alpha[j]减小，alpha[i]就增大
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                #为这两个alpha设置常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                #如果程序执行到这里还没有跳出，说明该alpha对已经成功改变。
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        #如果本次循环没有改变alpha对的值，就增加迭代次数再次for循环
        if (alphaPairsChanged == 0): iter += 1
        #如果改变了alpha对的值，就iter置零，继续运行。
        else: iter = 0
        print("iteration number: %d" % iter)
    #最终，只有在整个数据集上遍历maxIter次，而且不再发生任何alpha修改以后，程序才会退出while循环
    return b,alphas