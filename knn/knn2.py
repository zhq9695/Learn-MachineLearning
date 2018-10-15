# coding:utf-8
from numpy import *
import matplotlib.pyplot as plt
import operator

"""
约会网站案例
"""


# 将txt文中中的数据转换为矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # strip():
        #   移除字符串头尾的指定字符
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 均值归一化
def autoNorm(dataSet):
    # min(a):
    #   a=0 每列的最小值
    #   a=1 每行的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    meanVals = dataSet.mean(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(meanVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, meanVals


# 分类算法
def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # tile():
    #   在行方向上重复 intX，dataSetSize 次
    #   在列方向上重复 intX，1 次
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    # ** 表示平方
    sqDiffMat = diffMat ** 2
    # sum(axis=0) 表示每一列相加
    # sum(axis=1) 表示每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # argsort():
    #   按照数值从小到大，对数字的索引进行排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # {}.get(voteIlabel, 0):
        #   查找键值 voteIlabel，如果键值不存在则返回 0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # key=operator.itemgetter(1)
    #   获取对象第 1 个域的值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 测试分类算法
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, meanVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    correctCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d'
              % (classifierResult, datingLabels[i]))
        if classifierResult == datingLabels[i]:
            correctCount += 1.0
    print('the total accuracy is: %f' % (correctCount / float(numTestVecs)))


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    # add_subplot(321):
    #   将画图分割成 3 行 2 列，现在这个在从左到右从上到下第 1 个
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()
    datingClassTest()
