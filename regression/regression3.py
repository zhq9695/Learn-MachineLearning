# coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

"""
岭回归
"""


# 加载数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


# 岭回归测试函数
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # y进行均值化
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    # x进行均值归一化
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    # 测试lambda的次数
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

# 画lambda
# ax.plot(X) X为m*n时
# 横坐标为 m，[0, m-1]
# 纵坐标 [:, 1] 为第一条直线, [:, 2]为第二条直线....总共n条直线
def drawLambda(ridgeWeights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


if __name__ == '__main__':
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    print(ridgeWeights)
    drawLambda(ridgeWeights)