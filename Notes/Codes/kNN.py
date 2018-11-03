from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels
def classify0(inX,dataSet,labels,k):
    '''
    func : k nearest neighbors
    Parameters: 
    inX: 用于分类的输入向量
    dataSet：输入的训练样本集
    labels:标签向量
    k：选择最近邻居的数目
    '''
    #获取数据集大小
    dataSetSize = dataSet.shape[0] #shape是numpy下获取矩阵长度的函数
    #求出dataSet和inX扩展后矩阵的差矩阵
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #tile作用是在行方向和列方向上重复指定向量。
    #矩阵
    sqDiffMat = diffMat * 2 # * 代表乘法，**代表乘方
    sqDistances =sqDiffMat.sum(axis = 1)
    '''
	假如矩阵A是n*n的矩阵
	A.sum（）是计算矩阵A的每一个元素之和。
	A.sum(axis=0)是计算矩阵每一列元素相加之和。
	A.Sum(axis=1)是计算矩阵的每一行元素相加之和。
	'''
    distances = sqDistances**0.5#开根号
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
    	voteIlabel = labels[sortedDistIndicies[i]]
    	classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def main():
	group, labels = createDataSet()
	print(classify0([0,0],group,labels,3))
if __name__ =="__main__":
	main()

