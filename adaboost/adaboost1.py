# coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

"""
马病死亡案例
"""


# 加载数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 根据特征和阈值划分数据类别
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 建立单层决策树
# 第一层循环：遍历每一个特征
# 第二层循环：遍历每一个阈值
# 第三层循环：遍历小于阈值的是正类还是反类
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    # 阈值划分数量
    numSteps = 10.0
    # 最佳的决策树
    bestStump = {}
    # 最佳决策树的分类结果
    bestClasEst = mat(zeros((m, 1)))
    # 最小的加权错误率
    minError = inf
    # 第一层循环：遍历每一个特征
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 第二层循环：遍历每一个阈值
        for j in range(-1, int(numSteps) + 1):
            # 第三层循环：遍历小于阈值的是正类还是反类
            for inequal in ['lt', 'gt']:
                # 阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 根据特征和阈值划分数据
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 计算样本是否预测错误
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误率
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# 构建adaboost分类器
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化样本权重向量
    D = mat(ones((m, 1)) / m)
    # 加权的每个样本分类结果
    aggClassEst = mat(zeros((m, 1)))
    # 迭代
    for i in range(numIt):
        # 建立单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # 计算弱分类器权重
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        # 更新样本权重向量
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon)) / D.sum()
        # 更新每个样本的加权分类结果
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        # 计算当前加权的错误率
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


# 分类函数
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    # 加权的预测结果
    aggClassEst = mat(zeros((m, 1)))
    # 遍历每一个弱分类器
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)


# 画ROC曲线
def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    # 正类数量
    numPosClas = sum(array(classLabels) == 1.0)
    # 1/正类数量
    yStep = 1 / float(numPosClas)
    # 1/反类数量
    xStep = 1 / float(len(classLabels) - numPosClas)
    # 按照从小到大，索引排序
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 遍历排序后的索引
    # 表示属于正类的概率
    # 因排序，属于正类的概率越来越大，条件越来越苛刻
    # 由初始 TPF->1 FPR->1
    # 最终 TPR->0 FPR->0
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        # 更新cur
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    # 微积分算AUC面积
    # 对多个小长方块求面积之和
    # 小长方块的宽为 xStep
    # 每一次的长为 cur[1]_i
    # 即 cur[1]_1 * xStep + ... + cur[1]_n * xStep
    # 即 ySum * xStep
    print("the Area Under the Curve is: ", ySum * xStep)


if __name__ == '__main__':
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArr, aggClassEst = adaBoostTrainDS(datArr, labelArr, 50)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    pred = adaClassify(testArr, classifierArr)
    errArr = mat(ones((67, 1)))
    errNum = errArr[pred != mat(testLabelArr).T].sum()
    print(float(errNum) / len(testLabelArr))
    plotROC(aggClassEst.T, labelArr)
