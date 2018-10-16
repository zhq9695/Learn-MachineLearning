# coding:utf-8
from math import log
import operator
import pickle

"""
隐形眼镜案例
"""


# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    # 只选择第 axis 列的值为 value 的数据
    # 去除这个特征，取数据[:axis] 和 [axis+1:] 段
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历每一个特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历这个特征的所有特征值
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 判断这个子集占所有数据集的比例
            prob = len(subDataSet) / float(len(dataSet))
            # 新的信息熵 = 所有子集的信息熵乘以比例再求和
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 多数表决原则
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
# labels 为特征的标签
def createTree(dataSet, labels):
    # 获取当前数据集最后一列的类别信息
    classList = [example[-1] for example in dataSet]
    # 如果最后一列都是一种类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果当前数据集没有可划分的特征
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 获取最好的划分数据集特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 在特征标签中删除当前特征
    del (labels[bestFeat])
    # 获取这个特征的列，遍历此特征的所有特征值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 特征有几个取值，这个结点就有几个分支
        # 每个取值，都划分出子集，递归建树
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 分类函数
def classify(inputTree, featLabels, testVec):
    # 获取第一个特征
    firstStr = list(inputTree.keys())[0]
    # 获取这个特征下的键值对的值
    secondDict = inputTree[firstStr]
    # 获取这个特征的索引
    featIndex = featLabels.index(firstStr)
    # 遍历每一个分支
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # 判断当前分支下是否还有分支
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 存储树
def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


# 取出存储的树
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # storeTree(lensesTree, 'tree.txt')
    # lensesTree2 = grabTree('lensesTree.txt')
