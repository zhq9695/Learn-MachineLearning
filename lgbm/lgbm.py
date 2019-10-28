# -*- coding: utf-8 -*-
# @Time    : 2019/10/28 10:30
# @Author  : Zhanghanqi
# @FileName: lgbm.py
# @Software: PyCharm

from sklearn import datasets
import lightgbm as lgbm

data = datasets.load_digits()['data']
target = datasets.load_digits()['target']

clf = lgbm.LGBMClassifier().fit(data, target)
res = clf.predict(data)
print((res == target).sum() / target.shape[0])
