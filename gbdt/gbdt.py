# -*- coding: utf-8 -*-
# @Time    : 2019/10/28 8:48
# @Author  : Zhanghanqi
# @FileName: gbdt.py
# @Software: PyCharm

from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier

data = datasets.load_digits()['data']
target = datasets.load_digits()['target']

clf = GradientBoostingClassifier().fit(data, target)
