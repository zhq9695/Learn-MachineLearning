# coding:utf-8
from numpy import *

"""
病马死亡率案例
"""


# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 随机梯度上升
# stochastic gradient ascent
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    # 遍历每一个数据
    for i in range(m):
        # 数组对应元素相乘再相加
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进后的随机梯度上升
# 学习率随迭代次数减少
# 随机选择样本
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 学习率降低
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del (dataIndex[randIndex])
    return weights


# 分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


# 构件逻辑回归分类器，进行分类测试
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 遍历训练集
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 创建训练集的特征向量和标签
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 随机梯度上升求解参数
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    correctCount = 0
    numTestVec = 0.0
    # 遍历测试集
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) == int(currLine[21]):
            correctCount += 1
    accuracy = (float(correctCount) / numTestVec)
    print("the accuracy of this test is: %f" % accuracy)
    return accuracy


# 多次测试分类器
def multiTest():
    numTests = 10
    correctSum = 0.0
    for k in range(numTests):
        correctSum += colicTest()
    print("after %d iterations the average accuracy is: %f" % (numTests, correctSum / float(numTests)))


if __name__ == '__main__':
    multiTest()
