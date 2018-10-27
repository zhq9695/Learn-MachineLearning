# coding:utf-8
from numpy import *

"""
回归树
"""


# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将数据映射为浮点型
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 根据特征和特征值，二元分割一个数据集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 创建叶子结点时，采取所有剩余数据的标签的均值
def regLeaf(dataSet):
    return mean(dataSet[:, -1])


# 计算总方差
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# CART算法选择最佳划分点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 误差改善的最小要求
    tolS = ops[0]
    # 数据集大小的最小要求
    tolN = ops[1]
    # 如果当前数据集结果标签都是同一个值，则直接返回叶子节点
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    # 获取当前数据集的误差
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    # 一重循环遍历所有特征
    for featIndex in range(n - 1):
        # 二重循环遍历所有特征值
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            # 划分数据
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果划分后数据集太小则返回
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            # 计算新的误差
            newS = errType(mat0) + errType(mat1)
            # 如果新的误差小于当前最好的误差，则替换
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 遍历结束后，如果最佳的误差与遍历之前数据集的误差改善不大，则直接返回
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 划分两个数据子集
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果两个子集太小，则直接返回
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 递归创建树
# dataSet: 数据集
# leafType: 返回叶子节点的时候引用的函数
# errType: 误差计算引用的函数
# ops: 用户定义的标准值
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 选择最佳的划分点
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 当前为叶子节点
    if feat == None:
        return val
    # 记录当前的划分的特征和特征值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 划分两个数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归对两个子集创建子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 对树进行后剪枝
# 判断是否是子树
def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 对树进行后剪枝
# 递归获取当前节点的均值
# 在没有测试数据的时候，对节点进行塌陷处理
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 对树进行后剪枝，算法
def prune(tree, testData):
    # 如果没有测试数据了，则对树进行塌陷处理
    if shape(testData)[0] == 0:
        return getMean(tree)
    # 如果左节点或者右节点是树，则划分测试数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 如果左节点或者右节点是树，则递归后剪枝，直到叶子节点
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # 当前左节点和右节点都为叶子节点
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 划分测试数据集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算不合的误差
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        # 计算合并的误差
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        # 如果合并后误差小，则合并
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# 测试函数
# 返回叶子节点浮点类型值
def regTreeEval(model, inDat):
    return float(model)


# 预测函数
# inData是一条数据向量矩阵
def treeForeCast(tree, inData, modelEval=regTreeEval):
    # 叶子节点
    if not isTree(tree):
        return modelEval(tree, inData)
    # 选择左子树还是右子树
    if inData[tree['spInd']] > tree['spVal']:
        # 判断是否是树
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 预测函数测试
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    # myDat1 = loadDataSet('ex0.txt')
    # myMat1 = mat(myDat1)
    # tree1 = createTree(myMat1)
    # print(tree1)

    # myDat2 = loadDataSet('ex2.txt')
    # myMat2 = mat(myDat2)
    # tree2 = createTree(myMat2, ops=(0, 1))
    # myDat2Test = loadDataSet('ex2test.txt')
    # myMat2Test = mat(myDat2Test)
    # tree2 = prune(tree2, myMat2Test)
    # print(tree2)

    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
