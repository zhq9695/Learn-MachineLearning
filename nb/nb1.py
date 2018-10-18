# coding:utf-8
from numpy import *
import re

"""
垃圾邮件分类案例
"""


# 根据所有样本集合创建词汇表
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        # 集合求并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 通过词汇表，文本转换为向量
# 词集模型（每个词只记录是否有出现）
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


# 通过词汇表，文本转换为向量
# 词袋模型（每个词记录出现的次数）
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 训练NB分类器
def trainNB0(trainMatrix, trainCategory):
    # 文档数量
    numTrainDocs = len(trainMatrix)
    # 向量中特征（单词）的数量
    numWords = len(trainMatrix[0])
    # 类别1占总文档的比例
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 为防止再之后计算过程中某一个特征的概率为0，导致总的概率为0，不采用以下
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历每一个文本向量
    for i in range(numTrainDocs):
        # 如果向量属于类别1
        if trainCategory[i] == 1:
            # 通过向量，计算文档每个词汇的出现次数
            p1Num += trainMatrix[i]
            # 计算类别1中，单词的总数量
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1中，每个单词出现的次数/总单词数=每个单词的出现比例
    # 为防止数值太小，对其取Log
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 分类算法
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 因为 p1Vec 取过对数，log(x1)+...+log(xn)=log(x1*...*xn) 等于乘积
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 文件解析
def textParse(bigString):
    listOfTokens = re.split('\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 获取高频的前30个词汇
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    # 遍历词汇表中的每一个词
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    # 排序
    sortedFreq = sorted(freqDict.items(), key=lambda item: item[1], reverse=True)
    return sortedFreq[:30]


# 垃圾邮件测试
def spamTest():
    docList = []
    classList = []
    fullText = []
    # 遍历正的数据源和反的数据源
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词汇表
    vocabList = createVocabList(docList)
    # 获取高频词汇
    top30Words = calcMostFreq(vocabList, fullText)
    # 去除高频词汇，因高频词汇很可能是冗余词汇
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(50))
    testSet = []
    # 选择测试向量
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    # 创建训练集合
    for docIndex in trainingSet:
        # 将文本集合转换为文本向量矩阵
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # p0V: 类别0中，每个词汇出现的比例
    # p1V: 类别1中，每个词汇出现的比例
    # pSpam: 类别1的文本数量占总文本数量的比例
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    correctCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) == classList[docIndex]:
            correctCount += 1
    print('the correct rate is: ', float(correctCount) / len(testSet))
    return vocabList, p0V, p1V


# 获取最具表征性的词汇
def getTopWords(vocabList, p0V, p1V):
    top0 = []
    top1 = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            top0.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            top1.append((vocabList[i], p1V[i]))
    sorted0 = sorted(top0, key=lambda pair: pair[1], reverse=True)
    print("***** 0 *****")
    for item in sorted0:
        print(item[0])
    sorted1 = sorted(top1, key=lambda pair: pair[1], reverse=True)
    print('***** 1 *****')
    for item in sorted1:
        print(item[0])


if __name__ == '__main__':
    vocabList, p0V, p1V = spamTest()
    # getTopWords(vocabList, p0V, p1V)
