from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()  #sklearn.datasets.base.Bunch 类型
clf = tree.DecisionTreeClassifier()   # 创建DecisionTreeClassifier对象
clf = clf.fit(iris.data, iris.target)# 使用iris中的data属性和target属性来fit



import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
print(graph)