# coding:utf-8
from numpy import *
from numpy import linalg as la

"""
基于物品的协同过滤算法
"""


# 加载数据集
def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 欧式距离
def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


# 皮尔逊相关系数
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# 余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# 预测用户对于单个物品的评分
# dataMat: 数据矩阵
# user: 用户索引
# simMeas: 相似度函数
# item: 物品
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历每一个用户评价过的物品
    for j in range(n):
        # 用户对于该物品的评分
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # 获取同时评价了该物品和待预测物品的用户的索引
        overLap = nonzero(logical_and(dataMat[:, item].A > 0,
                                      dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            # 以这些用户对该物品的评分和这些用户对待预测物品的评分
            # 计算相似度
            similarity = simMeas(dataMat[overLap, item],
                                 dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 总相似度
        simTotal += similarity
        # 加权相似度
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 推荐算法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 获取用户未评分的物品的索引
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    # 遍历每一个物品，计算相似度
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


if __name__ == '__main__':
    myMat = mat(loadExData2())
    print(recommend(myMat, 0))
