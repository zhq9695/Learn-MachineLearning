import sys
from numpy import mat, mean, power

"""
分布式MapReduce
计算均值方差reducer
"""


def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)
mapperOut = [line.split('\t') for line in input]

cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj * float(instance[1])
    cumSumSq += nj * float(instance[2])

mean = cumVal / cumN
meanSq = cumSumSq / cumN

print("%d\t%f\t%f" % (cumN, mean, meanSq))
print("report: still alive", file=sys.stderr)
