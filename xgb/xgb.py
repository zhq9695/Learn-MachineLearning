# -*- coding: utf-8 -*-
# @Time    : 2019/10/28 9:54
# @Author  : Zhanghanqi
# @FileName: xgb.py
# @Software: PyCharm

from sklearn import datasets
import xgboost as xgb

data = datasets.load_digits()['data']
target = datasets.load_digits()['target']

clf2 = xgb.XGBClassifier().fit(data, target)
res = clf2.predict(data)
print((res == target).sum() / target.shape[0])
