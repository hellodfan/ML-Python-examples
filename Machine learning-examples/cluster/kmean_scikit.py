# coding:utf8
# kmeans 案例代码

from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

def create_data(centers, num=100, std=0.7):
    '''
        生成用于聚类的数据集
    :param centers: 聚类的中心点组成的数组。如果中心点是二维的，则产生的每个样本都是二维的。
    :param num: 总的样本数
    :param std: 生成数据簇中样本的标准差
    :return: 用于聚类的数据集。是一个元组，第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    '''
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return X, labels_true


def plot_data(*data):
    '''
    绘制用于聚类的数据集

    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    labels = np.unique(labels_true) # np.unique保留array中不同的数值，即保留所有labels
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = 'rgbyckm' # 每个簇的样本标记不同的颜色
    for i, label in enumerate(labels):
        position = labels_true == label # 选取出不同label的所有点的index
        ax.scatter(X[position, 0], X[position, 1], label="cluster %d"%label,
		color=colors[i%len(colors)], s=1) # 绘制数据 设置size=1

    ax.legend(loc="best", framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()



def test_Kmeans(*data):
    '''
    测试 KMeans 的用法

    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    clst=cluster.KMeans()  # 定义一个K-means 模型
    predicted_labels=clst.fit_predict(X)  # 训练模型并预测
    print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))  # 计算ARI指标  越大越好
    print("Sum center distance %s"%clst.inertia_)  # 样本距簇中心的距离和



def test_Kmeans_nclusters(*data):
    '''
    测试 KMeans 的聚类结果随 n_clusters 参数的影响

    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    ARIs=[]
    Distances=[]
    for num in nums:
        clst=cluster.KMeans(n_clusters=num)
        predicted_labels=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Distances.append(clst.inertia_)

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs,marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances,marker='o')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("inertia_")
    fig.suptitle("KMeans")
    plt.show()

def test_Kmeans_n_init(*data):
    '''
    测试 KMeans 的聚类结果随 n_init 和 init  参数的影响

    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    ARIs_k=[]
    Distances_k=[]
    ARIs_r=[]
    Distances_r=[]
    for num in nums:
            clst=cluster.KMeans(n_init=num,init='k-means++')  # 使用k-mean++策略
            predicted_labels=clst.fit_predict(X)
            ARIs_k.append(adjusted_rand_score(labels_true,predicted_labels))
            Distances_k.append(clst.inertia_)

            clst=cluster.KMeans(n_init=num,init='random')   # 使用random策略
            predicted_labels=clst.fit_predict(X)
            ARIs_r.append(adjusted_rand_score(labels_true,predicted_labels))
            Distances_r.append(clst.inertia_)

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs_k,marker="+",label="k-means++")
    ax.plot(nums,ARIs_r,marker="+",label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances_k,marker='o',label="k-means++")
    ax.plot(nums,Distances_r,marker='o',label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("inertia_")
    ax.legend(loc='best')

    fig.suptitle("KMeans")
    plt.show()



if __name__=='__main__':
    centers=[[1,1],[5,6],[1,10],[10,20]] # 用于产生聚类的中心点
    X,labels_true=create_data(centers,400,0.5) # 产生用于聚类的数据集
    #plot_data(X,labels_true)
    #test_Kmeans_nclusters(X,labels_true)
    test_Kmeans_n_init(X,labels_true)













