#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
"""
@Time     :2019/8/31 9:28

@Author   :Yuki

@FileName :test.py

@E-mail   :fujii20180311@foxmail.com
"""

from sklearn.neighbors import KNeighborsClassifier
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1.1], [2.1], [3.1]]))
print(neigh.score([[1.1], [2.1], [3.1]], [0, 1, 0]))
