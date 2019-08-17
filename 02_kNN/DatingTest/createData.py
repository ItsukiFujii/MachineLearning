#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
"""
@Time     :2019/7/31 16:11

@Author   :Yuki

@FileName :createData.py

@E-mail   :fujii20180311@foxmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy


def file2matric(file):
    fr = open(file)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        lineFromLine = line.split("\t")
        returnMat[index, :] = lineFromLine[0: 3]
        classLabelVector.append(int(lineFromLine[3]))
        index += 1
    return returnMat, classLabelVector


fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = file2matric('./datingTestSet2.txt')
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
plt.show()