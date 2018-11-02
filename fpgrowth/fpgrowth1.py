# coding:utf-8
from numpy import *

"""
FP-growth寻找频繁项集
"""


# 加载数据集
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


# 将数据集转换为set类型
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


# 树节点
class treeNode:
    # name: 节点名称
    # count: 出现次数
    # nodeLink: 节点链接
    # parent: 父节点
    # children: 子节点集
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    # 增加节点出现次数
    def inc(self, numOccur):
        self.count += numOccur

    # 打印此节点为树根的树
    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


# 创建FP树
def createTree(dataSet, minSup=1):
    headerTable = {}
    # 第一次遍历数据集
    # 获取单个元素的频率
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 去除不满足最小支持度的单个元素
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])
    # 频繁项集
    # freqItemSet: {'p', 'v', 'u', 'q', ...}
    freqItemSet = set(headerTable.keys())
    # 无频繁项就返回
    if len(freqItemSet) == 0:
        return None, None
    # 扩展头指针表
    # 添加指向每种类型第一个元素的指针（节点链接）
    # headerTable: {'j': [1, None], 'p': [2, None], 'r': [3, None], ...}
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 创建根节点
    retTree = treeNode('Null Set', 1, None)
    # 第二次遍历数据集
    # 构建FP树
    for tranSet, count in dataSet.items():
        # tranSet: frozenset({'h', 'p', 'z', 'j', 'r'})
        # count: 1
        localD = {}
        # 如果单个元素是频繁项，则加入localD列表
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        # localD: {'r': 3, 'j': 1, 'z': 5, 'h': 1, 'p': 2}
        if len(localD) > 0:
            # 排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # 更新FP树
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


# 更新FP树函数
def updateTree(items, inTree, headerTable, count):
    # 判断排序后列表的第一个元素是否已经是根节点的子节点
    if items[0] in inTree.children:
        # 添加出现次数
        inTree.children[items[0]].inc(count)
    else:
        # 创建根节点的子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针表的节点链接
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 列表元素长度大于1
    # 递归调用更新FP树函数
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


# 更新头指针表的节点链接的函数
def updateHeader(nodeToTest, targetNode):
    # 将元素放在指针链表的最后
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


# 寻找节点basePat的所有前缀路径
# treeNode: 头节点表的basePat的指针指向元素
def findPrefixPath(basePat, treeNode):
    condPats = {}
    # 有指向的元素
    while treeNode != None:
        prefixPath = []
        # 回溯父节点，寻找前缀路径
        # prefixPath: ['r', 't', 'x', 'z']
        ascendTree(treeNode, prefixPath)
        # 路径长度大于1，不是单个元素
        if len(prefixPath) > 1:
            # 添加进condPats，记录路径的出现次数
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 继续寻找basePat为结尾的前缀路径
        treeNode = treeNode.nodeLink
    # condPats: {frozenset({'z'}): 1, frozenset({'s', 'x'}): 1, frozenset({'x', 'z'}): 1}
    return condPats


# 单个节点回溯，寻找前缀路径
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


# 根据FP树寻找频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 头指针表排序
    # bigL: ['h', 'j', 'u', 'v', 'w',...]
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        print('finalFrequent Item: ', newFreqSet)
        freqItemList.append(newFreqSet)
        # 以basePat为节点的所有前缀路径
        # condPattBases: {frozenset({'z', 'r', 'p'}): 1, ...}
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('condPattBases :', basePat, condPattBases)
        # 以当前元素的所有前缀路径，创建条件FP树
        myCondTree, myHead = createTree(condPattBases, minSup)
        print('head from conditional tree: ', myHead)
        # 根据条件FP树和条件头指针表，递归创建下一个条件FP树
        if myHead != None:
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


if __name__ == '__main__':
    # simpDat = loadSimpDat()
    # initSet = createInitSet(simpDat)
    # retTree, headerTable = createTree(initSet)
    # freqItems = []
    # mineTree(retTree, headerTable, 3, set([]), freqItems)
    # print(freqItems)

    parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parsedDat)
    myFPtree, myHeaderTab = createTree(initSet, 100000)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    print(myFreqList)
