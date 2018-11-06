# coding:utf-8
from numpy import *
from numpy import linalg as la

"""
基于用户的协同过滤算法
"""


# 加载数据集
def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [1, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
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
    m = shape(dataMat)[0]
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历每一个评价过该物品的用户
    for i in range(m):
        # 用户对于该物品的评分
        userRating = dataMat[i, item]
        if userRating == 0:
            continue
        # 获取待预测用户和该用户都评价过的商品列表
        overLap = nonzero(logical_and(dataMat[user, :].A > 0,
                                      dataMat[i, :].A > 0))[1]
        if len(overLap) == 0:
            similarity = 0
        else:
            # 以该用户对这些物品的评分和待预测用户对这些物品的评分
            # 计算相似度
            similarity = simMeas(dataMat[user, overLap].T,
                                 dataMat[i, overLap].T)
        print('the %d and %d similarity is: %f' % (user, i, similarity))
        # 总相似度
        simTotal += similarity
        # 加权相似度
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 推荐算法
def recommend(dataMat, item, N=3, simMeas=cosSim, estMethod=standEst):
    # 获取未对该物品评分的用户
    unratedItems = nonzero(dataMat[:, item].A == 0)[0]
    if len(unratedItems) == 0:
        return 'everyone rated item'
    itemScores = []
    # 遍历每一个用户，计算相似度
    for user in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((user, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


if __name__ == '__main__':
    myMat = mat(loadExData2())
    print(recommend(myMat, 0))
