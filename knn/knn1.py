# coding:utf-8
from numpy import *
import operator

"""
简单案例
"""


# 创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


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


if __name__ == '__main__':
    group, labels = createDataSet()
    intX = [0, 0]
    k = 3
    clasifierResult = classify0(intX, group, labels, k)
    print(clasifierResult)
