# -*- coding: utf-8 -*-
#  使用EM算法解算GGM  EM算法采用scikit-learn包提供的api
#  数据集：《机器学习》--西瓜数据4.0   :文件watermelon4.txt

from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np


# 预处理数据
def loadData(filename):
    dataSet = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet


def test_GMM(dataMat, components=3,iter = 100,cov_type="full"):
    clst = mixture.GaussianMixture(n_components=n_components,max_iter=iter,covariance_type=cov_type)
    clst.fit(dataMat)
    predicted_labels =clst.predict(dataMat)
    return clst.means_,predicted_labels     # clst.means_返回均值



def showCluster(dataMat, k, centroids, clusterAssment):
    numSamples, dim = dataMat.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i])
        plt.plot(dataMat[i, 0], dataMat[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


if __name__=="__main__":
    dataMat = np.mat(loadData('watermelon4.txt'))
    n_components = 3
    iter=100
    cov_types = ['spherical', 'tied', 'diag', 'full']
    centroids,labels = test_GMM(dataMat,n_components,iter,cov_types[3])
    showCluster(dataMat, n_components, centroids, labels)  # 这里labels维度改变了，注意修改showCluster方法