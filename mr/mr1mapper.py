# coding:utf-8
import sys
from numpy import mat, mean, power

"""
分布式MapReduce
计算均值方差mapper
"""


def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = mat(input)
sqInput = power(input, 2)

print("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)))
print("report: still alive", file=sys.stderr)
