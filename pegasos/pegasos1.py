# coding:utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

"""
Pegasos: 支持向量机的原始估计子梯度求解器
primal estimated sub-gradient solver for SVM
"""


# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# 线性预测
def predict(w, x):
    return w * x.T


# pegasos 随机梯度下降
# 1次只选择1个样本
def seqPegasos(dataSet, labels, lam, T):
    m, n = shape(dataSet)
    w = zeros(n)
    # 迭代T次
    for t in range(1, T + 1):
        i = random.randint(m)
        eta = 1.0 / (lam * t)
        p = predict(w, dataSet[i, :])
        # 不满足SVM要求
        if labels[i] * p < 1:
            w = (1.0 - 1 / t) * w + eta * labels[i] * dataSet[i, :]
        else:
            w = (1.0 - 1 / t) * w
    return w


# pegasos 随机梯度下降
# 1次只选择k个样本
def batchPegasos(dataSet, labels, lam, T, k):
    m, n = shape(dataSet)
    w = zeros(n)
    dataIndex = arange(m)
    # 迭代T次
    for t in range(1, T + 1):
        wDelta = mat(zeros(n))
        eta = 1.0 / (lam * t)
        random.shuffle(dataIndex)
        for j in range(k):
            i = dataIndex[j]
            p = predict(w, dataSet[i, :])
            # 不满足SVM要求
            if labels[i] * p < 1:
                wDelta += labels[i] * dataSet[i, :].A
        # 取k次的平均
        w = (1.0 - 1 / t) * w + (eta / k) * wDelta
    return w


if __name__ == '__main__':
    datArr, labelList = loadDataSet('testSet.txt')
    datMat = mat(datArr)
    # finalWs = seqPegasos(datMat, labelList, lam=2, T=5000)
    finalWs = batchPegasos(datMat, labelList, lam=2, T=50, k=100)
    print(finalWs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = []
    y1 = []
    xm1 = []
    ym1 = []
    # 区分两个类别
    for i in range(len(labelList)):
        if labelList[i] == 1.0:
            x1.append(datMat[i, 0])
            y1.append(datMat[i, 1])
        else:
            xm1.append(datMat[i, 0])
            ym1.append(datMat[i, 1])
    ax.scatter(x1, y1, marker='s', s=90)
    ax.scatter(xm1, ym1, marker='o', s=50, c='red')
    x = arange(-6.0, 8.0, 0.1)
    y = (-finalWs[0, 0] * x - 0) / finalWs[0, 1]
    ax.plot(x, y, 'g-')
    ax.axis([-6, 8, -4, 5])
    plt.show()
