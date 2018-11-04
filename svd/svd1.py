# coding:utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

"""
svd降维
矩阵分解，利用低维空间特征表示高维数据
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


# svd矩阵分解，低维空间还原回原矩阵
def svd2():
    Data = mat([[0, 0, 0, 2, 2],
                [0, 0, 0, 3, 3],
                [0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0],
                [2, 2, 2, 0, 0],
                [5, 5, 5, 0, 0],
                [1, 1, 1, 0, 0]])

    U, sigma, VT = linalg.svd(Data)
    # U: 7*7
    # sigma: 7*5
    # VT: 5*5
    print(U.shape, sigma.shape, VT.shape)
    # 奇异值
    print(sigma)
    # 使用前两个特征近似表示原始矩阵
    print(mat(U[:, :2]) * mat(multiply(eye(2), sigma[:2])) * VT[:2, :])


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

    svd2()
