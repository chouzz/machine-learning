import numpy as np
import operator
import matplotlib.pyplot as plt

dataSet = [['sunny', 'hot', 'high', 'FALSE', 'no'],
           ['sunny', 'hot', 'high', 'TRUE', 'no'],
           ['overcast', 'hot', 'high', 'FALSE', 'yes'],
           ['rainy', 'mild', 'high', 'FALSE', 'yes'],
           ['rainy', 'cool', 'normal', 'FALSE', 'yes'],
           ['rainy', 'cool', 'normal', 'TRUE', 'no'],
           ['overcast', 'cool', 'normal', 'TRUE', 'yes'],
           ['sunny', 'mild', 'high', 'FALSE', 'no'],
           ['sunny', 'cool', 'normal', 'FALSE', 'yes'],
           ['rainy', 'mild', 'normal', 'FALSE', 'yes'],
           ['sunny', 'mild', 'normal', 'TRUE', 'yes'],
           ['overcast', 'mild', 'high', 'TRUE', 'yes'],
           ['overcast', 'hot', 'normal', 'FALSE', 'yes'],
           ['rainy', 'mild', 'high', 'TRUE', 'no']]
labels = ['outlook','temperature','humidity','windy','play']

def calcentropy(data):
    numentropy = len(data)
    labelCounts = {}
    for featVec in data:  # 数据集的每一行
        currentlabel = featVec[-1]  # 每一行的最后一个类别
        if currentlabel not in labelCounts.keys():  # 如果当前标签不在字典的关键字中
            labelCounts[currentlabel] = 0  # 让当前标签为0，实际上是增加字典的关键字
        labelCounts[currentlabel] += 1  # 如果在字典里，就增加1，实际上是统计每个标签出现的次数

    shannonEnt = 0  # 香农熵，即最后的返回结果
    for key in labelCounts:     # 遍历labelCount中的每一个关键词
        prob = labelCounts[key] / numentropy    # 用关键词的个数除以总个数得到概率
        shannonEnt -= prob * np.log2(prob)      # 求信息熵，即香农熵
    return shannonEnt


def splitData(data, axis, value):       # 划分数据集
    retData = []
    for featVec in data:    # 用featVec表示每一个样本
        if featVec[axis] == value:      # 如果value的值等于里面样本的特征
            reduceFeat = featVec[:axis]     # 用reduceFeat补齐这个特征之前的所有特征
            reduceFeat.extend(featVec[axis + 1:])       # 补齐这个特征之后的所有特征
            retData.append(reduceFeat)      # 用retData来表示去掉这个特征的最终特征
    return retData


def chooseBestFeatSplit(data):
    numFeats = len(data[0]) - 1     # 特征的数目，减去最后一个特征
    bestGain = 0
    for i in range(numFeats):       # i：0-3，在本例中只有4个特征
        featlist = [example[i] for example in data]     # 列表解析取出data中的一列数据
        uniqueVals = set(featlist)          # 转变为集合
        newEntropy = 0
        for value in uniqueVals:            # 遍历每个特征内所有集合的元素，对于i=0是，uniqueVals={summy,rainy,overcast}
            subData = splitData(data, i, value)      # 对第一个特征中rainy划分数据集
            prob = len(subData) / float(len(data))
            newEntropy += prob * calcentropy(subData)   # 求出条件熵
        infoGain = baseGain - newEntropy                # 计算信息增益
        if infoGain > bestGain:                         # 取出最大的信息增益
            bestGain = infoGain
            bestfeat = i
    return bestfeat                                     # 返回当信息增益最大是的特征类别

def majorityCnt(classList):     # classList 为一个列表
    classCount = {}
    for vote in classList:      # 遍历列表的每个元素
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1   # 用字典统计列表中的元素出现次数，以键、值来表示
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reversed=True)
    # key关键字表达的是根据第二个域来排序，即第一次循环根据‘yes’‘no’的个数来排序，且为降序
    return sortedClassCount[0][0]   # 返回原列表中出现次数最多的特征标签


def creatTree(data, labels):
    classList = [example[-1] for example in data]       # 遍历数据集最后一列
    if classList.count(classList[0]) == len(classList):     # 类别中的概率是否确定，即概率为1时，提前到达叶节点
        return classList[0]                             # 返回该类别
    if len(data[0]) == 1:       # 是否只剩下一个类别，到达最终的叶节点
        return majorityCnt(classList)
    bestFeature = chooseBestFeatSplit(data)         # 选择最好的特征
    bestFeatureLabel = labels[bestFeature]          # 最好的特征表现
    myTree = {bestFeatureLabel:{}}                  # 创建树
    del(labels[bestFeature])                        # 删除已经计算过的特征
    featValues = [example[bestFeature] for example in data]     # 遍历根据最佳特征分割完成后的类别中的属性
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sublabels = labels[:]
        myTree[bestFeatureLabel][value] = creatTree(splitData(data,bestFeature,value),sublabels)
    return myTree


def getNumLeafs(myTree):
    numLeafs = 0
    # firstStr = myTree.keys()[0] # 版本问题，无法使用
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    # firstStr = myTree.keys()[0]
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    # firstStr = myTree.keys()[0]  # the text label for this node should be this
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]

if __name__ == '__main__':
    baseGain = calcentropy(dataSet)
    myTree = creatTree(dataSet,labels)
    print(myTree)

    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arrowstyle="<-")


    createPlot(myTree)

