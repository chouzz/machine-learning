from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import graphviz

datamat = np.loadtxt('data.csv',delimiter=',',header="#1,#2,#3,#4")

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
print(graph)