# coding:utf-8
from numpy import *

"""
原始k-means
"""


# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 随机生成k个质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    # K个质心的向量
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


# k-means算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    # 每个数据对应的簇和距离
    clusterAssment = mat(zeros((m, 2)))
    # 随机生成k个质心
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历每一个数据
        for i in range(m):
            minDist = inf
            minIndex = -1
            # 遍历每一个质心
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 当前数据的质心改变
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 记录当前质心和距离
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 重新计算每个质心的位置
        for cent in range(k):
            # 获取属于当前簇的数据
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 按列求平均，重新计算质心位置
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


if __name__ == '__main__':
    datMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clusterAssing = kMeans(datMat, 4)
    print(myCentroids)