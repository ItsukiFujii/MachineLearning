#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import operator
import sklearn
"""
@Time     :2019/8/7 14:05

@Author   :Yuki

@FileName :kNN_test02.py

@E-mail   :fujii20180311@foxmail.com
"""

def classify0(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，计算出距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]
'''
函数说明：打开并解析文件，并对数据进行分类

Parameters：
    filename：文件名

Returns：
    returnMat：特征矩阵
    classLabelVector：标签的分类
'''
def file2Matrix(filename):
    #打开文件
    fr = open(filename)
    #读入文件所有行内容
    arrayOLines = fr.readlines()
    #获取数据行数
    numberOfLines = len(arrayOLines)
    #返回ndarray矩阵，numberOfLines行，3列，填充0
    returnMat = np.zeros((numberOfLines, 3))
    #记录标签类型
    classLabelVector = []
    #行索引
    index = 0
    for line in arrayOLines:
        #默认删除字符串首和尾的空白字符，'\n','\t','\r',' '等
        line = line.strip()
        #使用\t对字符串进行分片
        listFromLine = line.split('\t')
        #取出listFromLine的前3行存入returnMat的一行中
        returnMat[index, :] = listFromLine[0: 3]
        #记录数据标签
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
'''
函数说明：对数据进行归一化处理

Parameters:
    dataset:特征矩阵

Return:
    normalDataSet:归一化的特征矩阵
    ranges:数据范围
    minVals:数据最小值
'''
def autoNorm(dataSet):
    #获取数据案列算出最小最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #算出最大值和最小值的范围
    ranges  = maxVals-minVals
    #以dataSet数组形状创建一个值都为0的数组
    normalDataSet = np.zeros(dataSet.shape)
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normalDataSet = dataSet-np.tile(minVals, (m, 1))
    #差值除以数据值范围
    normalDataSet = normalDataSet / np.tile(ranges, (m, 1))
    return normalDataSet, ranges, minVals


'''
函数说明：数据可视化

Parameters:
    datingDataMat:特征矩阵
    datingLabels:分类Label

Returns:
    无
'''
def showdatas(datingDataMat, datingLabels):
    #将画布分成2行2列，画布大小为(13, 8),不共享x轴和y轴
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    #获取标签行数
    # numberOfLabels = len(datingLabels)
    labelsColors = []
    for i in datingLabels:
        if i == 1:
            labelsColors.append('black')
        elif i == 2:
            labelsColors.append('orange')
        else:
            labelsColors.append('red')
    #画出散点图，以datingDataMat的第一列（飞行里程），第二列（玩游戏）数据画散点图，散点大小为15，透明度0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=labelsColors, s=15, alpha=0.5)
    #设置标题，x轴label，y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数和玩视频游所消耗时间百分比', fontproperties='SimHei')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', fontproperties='SimHei')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', fontproperties='SimHei')
    d = dict(size=12, weight='bold', color='pink')
    plt.setp(axs0_title_text, **d)
    plt.setp(axs0_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=12, weight='bold', color='black')

    #画出散点图，以datingDataMat的第一列（飞行里程），第三列（冰激凌）数据画散点图，散点大小为15，透明度0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=labelsColors, s=15, alpha=0.5)
    #设置标题，x轴label，y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数和每周消费的冰激凌公升数', fontproperties='SimHei')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', fontproperties='SimHei')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激凌公升数', fontproperties='SimHei')
    plt.setp(axs1_title_text, size=12, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=12, weight='bold', color='black')

    #画出散点图，以datingDataMat的第二列（游戏），第三列（冰激凌）数据画散点图，散点大小为15，透明度0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=labelsColors, s=15, alpha=0.5)
    #设置标题，x轴label，y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所耗时间占比和每周消费的冰激凌公升数', fontproperties='SimHei')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所耗时间占比', fontproperties='SimHei')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激凌公升数', fontproperties='SimHei')
    plt.setp(axs2_title_text, size=12, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=12, weight='bold', color='black')

    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # handles, labels = axs[0][0].get_legend_handles_labels()
    # print(handles)
    plt.show()
'''
函数说明:分类器测试函数

Parameters:
    fileName:文件名

Returns:
    无
'''
def datingClassTest(fileName):
    #调用file2Matrix()得到返回的特征数据和标签
    datingDataMat, datingLabels = file2Matrix(fileName)
    #取出所有数据的前10%
    hoRatio = 0.1
    #数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获取归一化特征矩阵的行数
    m = normMat.shape[0]
    #10%的测试数据
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集，后m-numTestVecs作为个数据作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:, :], datingLabels[numTestVecs:], 4)
        print("分类类型:%d\t真实类型:%d" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("分类的正确率为:%.2f%%" % ((m-errorCount)/m*100))
    return
if __name__ == '__main__':
    datingDataMat, datingLabels = file2Matrix("./datingTestSet2.txt")
    showdatas(datingDataMat, datingLabels)



