#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import numpy as np
import operator
"""
@Time     :2019/8/7 12:54

@Author   :Yuki

@FileName :KNN_test01.py

@E-mail   :fujii20180311@foxmail.com
"""

'''
函数说明：创建数据集

Parameters：
    无
    
Returns：
    group：数据集
    labels：标签集
'''

def createDataSet():
    group = np.array([[3, 104], [2, 100], [101, 10], [99, 5]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels
'''
函数说明：kNN分类

Parameters：
    inX：测试数据
    dataSet：训练集
    labels：标签集
    k：kNN参数，选择距离最近的k个点
'''
def classify0(inX, dataSet, labels, k):
    #训练集的行数
    dataSetSize = dataSet.shape[0]
    #横向扩展dataSetSize次，纵向扩展1次
    diffMat = np.tile(inX, (dataSetSize, 1))-dataSet
    #二维特征相减后，求平方
    sqDiffMat = diffMat**2
    #sum()函数，列方向上相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开根号求得距离值
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后得索引值
    sortedDistIndicies = distances.argsort()
    #记录类别次数得字典
    classCount = {}
    for i in range(k):
        #取出前k个元素得类别
        voteIlabel = labels[sortedDistIndicies[i]]
        #dict.get(key, default)函数，返回指定key的value，否则返回default
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    #items,将字典键值对封装成元组
    #operator.itemgetter(0),对字典的键进行排序
    #operator.itemgetter(1),对字典的值进行排序
    #reverse降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回出现次数最多的类型
    return sortedClassCount[0][0]
if __name__ == '__main__':
    #创建数据集，和分类标签
    group, labels = createDataSet()
    #创建测试集
    test = [101, 20]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    print(test_class)


