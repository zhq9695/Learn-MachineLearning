# coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

"""
简单案例
"""


# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升求解J(\theta)
# Batch gradient ascent
def gradAscent(dataMatIn, classLabels):
    # 转换为矩阵
    dataMatrix = mat(dataMatIn)
    # 转换为矩阵后转置，表示为列向量
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    # \theta 表示为列向量
    weights = ones((n, 1))
    # 梯度上升迭代
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 加载数据集
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        # 去除头尾空字符，按照空格分割字符串
        lineArr = line.strip().split()
        # 添加偏置位 w_0(\theta_0) 相乘的 x_0 = 1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# 画出数据的分界线
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # 遍历每一条数据，根据类别将x1, x2分别插入不同的List中
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = array(arange(-3.0, 3.0, 0.1))
    # x=x1, y=x2 表示的是 w0x0+w1x1+w2x2 = 0 的直线
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    # getA():
    #   将矩阵转换为ndarray
    plotBestFit(weights.getA())
