# coding:utf-8
from numpy import *
from numpy import linalg as la

"""
svd降维实现矩阵压缩
"""


# 输出矩阵
def printMat(inMat, thresh=0.8):
    for i in range(32):
        s = ''
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                s += '1'
            else:
                s += '0'
        print(s)


# 降维压缩矩阵
def imgCompress(numSV=2, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    # 打印原始矩阵
    print("****original matrix******")
    printMat(myMat, thresh)
    # svd压缩矩阵
    # U: m*m
    # sigma: m*n
    # VT: n*n
    U, Sigma, VT = la.svd(myMat)
    # 构建sigma矩阵
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    # 利用压缩后的矩阵还原原矩阵
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    # 打印还原后的矩阵
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':
    imgCompress(2)
