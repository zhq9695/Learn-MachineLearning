# coding:utf-8
from numpy import *

"""
apriori算法发现频繁项集和关联规则
"""


# 加载数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 创建初始只包含单个元素的数据项集
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))


# 扫描数据集
# 计算当前数据项集中满足最小支持度的项集
def scanD(D, Ck, minSupport):
    ssCnt = {}
    # 遍历每一条数据
    for tid in D:
        # 遍历每一个项集
        for can in Ck:
            # 项集是数据的一个子集
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 遍历每一个项集，计算支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
    return retList, supportData


# 根据满足最小支持度的项集
# 计算项集的组合
def aprioriGen(Lk, k):  # creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 当0~k-2个项相同的时候
            # 合并可以得到长度为k的项，且不会重复
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


# apriori算法，生成频繁项集
# 此处并没有不计算那些项集为非频繁的超集，依旧按照原始计算
def apriori(dataSet, minSupport=0.5):
    # 长度为1的项集
    C1 = createC1(dataSet)
    # 数据集D
    D = list(map(set, dataSet))
    # 满足最小支持度的项集l1，支持度supportData
    L1, supportData = scanD(D, C1, minSupport)
    # 构建列表
    L = [L1]
    k = 2
    # 当当前满足支持度的项集个数大于0时，继续计算
    while (len(L[k - 2]) > 0):
        # 计算当前满足支持度的项集的组合
        Ck = aprioriGen(L[k - 2], k)
        # 计算组合后，满足最小支持度的项集和支持度
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        # 将满足最小支持度的项集添加进L
        L.append(Lk)
        k += 1
    return L, supportData


# 规则分析
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    # 只判断项集中元素大于1的情况，因对其进行拆分
    for i in range(1, len(L)):
        # 当前每个项集的长度为i+1，遍历每个项集
        for freqSet in L[i]:
            # freqSet: frozenset({1, 3})
            # H1: [frozenset({1}), frozenset({3})]
            H1 = [frozenset([item]) for item in freqSet]
            # 项集长度大于2
            if (i > 1):
                # 后件的长度为1，返回大于最小可信度的hmp1
                # 运用大于最小可行度的后件，再进行组合，生成更长的后件，分级判断
                #
                ######################################################
                # 这部分与书中不同，本人认为书中有错误，缺少下面第一二行  #
                # 按照书中，缺少判断 前件长度>1且后件长度=1的情况        #
                # 希望可以得到指正                                    #
                ######################################################
                #
                Hmp1 = calcConf(freqSet, H1, supportData, bigRuleList, minConf)
                if (len(Hmp1) > 1):
                    rulesFromConseq(freqSet, Hmp1, supportData, bigRuleList, minConf)
            else:
                # 项集长度只有2
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 计算以H中的一个为前件，一个为后件的可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        # 计算可信度
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# 分级计算长度大于2的项集的规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    # 单个频繁项集的长度
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        # 原先长度为m，生成长度为m+1的
        # H: [frozenset({2}), frozenset({3}), frozenset({5})]
        # Hmp1: [frozenset({2, 3}), frozenset({2, 5}), frozenset({3, 5})]
        Hmp1 = aprioriGen(H, m + 1)
        # 将Hmp1中的每一个作为后件，计算可信度
        # 返回大于最小可信度的后件，用作下一次调用此函数时，组合成新的后件
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            # 还可进一步扩大后件的长度
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


if __name__ == '__main__':
    # dataSet = loadDataSet()
    # L, suppData = apriori(dataSet)
    # rules = generateRules(L, suppData)

    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L, suppData = apriori(mushDatSet, minSupport=0.3)
    for item in L[1] + L[2]:
        if item.intersection('2'):
            print(item)
