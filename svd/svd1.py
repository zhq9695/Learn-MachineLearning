# coding:utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

"""
svd降维
"""


# 加载数据集
def loadDataSet(fileName, delim):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


# SVD降维
def svd(dataMat, topNfeat=9999999):
    # 均值归一化
    meanVals = dataMat.mean(axis=0)
    maxVals = dataMat.max(axis=0)
    minVals = dataMat.min(axis=0)
    meanRemoved = (dataMat - meanVals) / (maxVals - minVals)
    # 协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)
    # 奇异值分解
    U, sigma, VT = linalg.svd(covMat)
    # 降维
    lowDDataMat = meanRemoved * U[:, :topNfeat]
    # 映射回高维空间中，不过不是原始值，而是低维空间点对应的高维空间位置
    reconMat = multiply((lowDDataMat * U[:, :topNfeat].T), (maxVals - minVals)) + meanVals
    return lowDDataMat, reconMat


if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt', '\t')
    lowDMat, reconMat = svd(dataMat, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0],
               marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0],
               marker='o', s=50, c='red')
    plt.show()
