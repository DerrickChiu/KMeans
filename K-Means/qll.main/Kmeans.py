# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

import sys
sys.path.append('../qll.tool')

from function import *


create(500,'F://exam/test.txt')
dataMat1 = file2matrix('F://exam/test.txt')

def k_Means(k,type,dataMat):      #传统kmeans聚类主函数
    distlist = []
    minDist = 0
    m = shape(dataMat)[0]
    # k = 4
    ClustDist = mat(zeros((m, 2)))
    centers = randCenters(dataMat, k)  #随机生成每一类的中心点
    flag = True
    counter = []

    while flag:    #主循环：只有当每类的中心点不再改变时才认为聚类完成
        flag = False
        for i in range(m):  #循环遍历每条数据将其归类到最近的中心点所在的类
            if type == 'eclu':  #采用欧式距离
                distlist = [ecluddistance(centers[j, :], dataMat[i, :]) for j in range(k)]
                minDist = min(distlist)
            if type == 'cos':   #采用夹角余弦
                distlist = [cosdistance(centers[j,:],dataMat[i,:]) for j in range(k) ]
                minDist = max(distlist)

            minIndex = distlist.index(minDist)  

            if ClustDist[i, 0] != minIndex:   #如果有数据点改变了分类就要继续主循环
                flag = True

            ClustDist[i, :] = minIndex, minDist
        for cent in range(k):
            ptsInCluster = dataMat[nonzero(ClustDist[:, 0].A == cent)[0]]
            centers[cent, :] = mean(ptsInCluster, axis=0)
    return centers,ClustDist

# centers,ClustDist = k_Means(2,'cos');
# showplt(plt,ClustDist,centers,dataMat)

def k_means_2split(k,type):   #二分kmeans聚类主函数
    m = shape(dataMat1)[0]
    centroid0 = mean(dataMat1,axis=0).tolist()[0]
    centList = [centroid0]
    ClusDist = mat(zeros((m,2)))
    for j in range(m):
        ClusDist[j,1] = ecluddistance(centroid0,dataMat1[j,:])**2
    while(len(centList) < k):
        lowestSSE = inf
        
        '''
        #二分kmeans聚类是传统kmeans算法的改进，
        会提高算法的准确率，但要以时间为代价
        即通过增加时间复杂度来换取准确率
        '''
        for i in range(len(centList)):  
            ptsInCurrCluster = dataMat1[nonzero(ClusDist[:,0].A==i)[0],:]
            centroidMat,splitClustAss = k_Means(2,type,ptsInCurrCluster)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(ClusDist[nonzero(ClusDist[:,0].A!=i)[0],1])
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNotSplit + sseSplit
        

        '''
        #每次选取一个类进行二分,策略如下：
        对已存在的每个类试划分，选取能够最大限度降低误差平方和的类进行正式划分
        '''
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        ClusDist[nonzero(ClusDist[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),mat(ClusDist)

centers,ClustDist = k_means_2split(4,'eclu')  #为二分kmeans聚类模型生成数据点并执行二分kmeans聚类算法
showplt(plt,centers,ClustDist,dataMat1)  #显示二分kmeans聚类效果
centers1,ClustDist1 = k_Means(4,'eclu',dataMat1)   #为传统kmeans聚类模型生成数据点并执行传统kmeans聚类算法
showplt(plt,centers1,ClustDist1,dataMat1)   #显示传统kmeans聚类效果
plt.show()