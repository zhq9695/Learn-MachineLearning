# coding:utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

"""
二分k-means
"""


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


# 二分k-means
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    # 所有数据属于的簇和与簇质心的距离
    clusterAssment = mat(zeros((m, 2)))
    # 初始化第一个簇
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 簇列表
    centList = [centroid0]
    # 计算每一个数据与当前唯一簇的距离
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    # 循环执行二分，直到簇的数量达到k
    while (len(centList) < k):
        lowestSSE = inf
        # 遍历每一个簇
        for i in range(len(centList)):
            # 获取属于当前簇的数据
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 进行2-means
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算误差平方和
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            # 如果小于最小误差，则替换
            if (sseSplit + sseNotSplit) < lowestSSE:
                # 最佳划分簇
                bestCentToSplit = i
                # 新的两个簇向量
                bestNewCents = centroidMat
                # 原先属于最佳划分簇的数据，当前属于的新簇和距离
                bestClustAss = splitClustAss.copy()
                # 最低的误差平方和
                lowestSSE = sseSplit + sseNotSplit
        # 将划分后，数据属于新簇的簇索引
        # 由0, 1改为bestCentToSplit和len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 修改被划分的簇的质心，添加新的簇
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        # 取属于被划分簇的那些数据，修改这些数据的新的簇和到簇质心的距离
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = \
            bestClustAss  # reassign new clusters, and SSE
    return mat(centList), clusterAssment


# 球面余弦定理
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


# 聚类，画图
def clusterClubs(numClust=5):
    datList = []
    # 加载数据集
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    # 二分k-means
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    # 创建画板
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    # 创建底图
    # 每一个axes都是一个独立的图层
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    # 创建数据图层
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    # 画每一个簇的数据
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0],
                    marker=markerStyle, s=90)
    # 画簇的质心
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    clusterClubs(5)
