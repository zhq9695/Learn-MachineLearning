# coding:utf-8
from numpy import *
import operator
from os import listdir

"""
手写识别案例
"""


# 将01文本表示的图像转换为向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


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
def handwritingClassTest():
    hwLabels = []
    # 读取目录下文件列表
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    testFileList = listdir('testDigits')
    correctCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d'
              % (classifierResult, classNumStr))
        if classifierResult == classNumStr:
            correctCount += 1.0
    print('the total accuracy is: %f' % (correctCount / float(mTest)))


if __name__ == '__main__':
    handwritingClassTest()
