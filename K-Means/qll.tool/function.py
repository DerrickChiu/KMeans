# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:41:50 2020

@author: Administrator
"""

import numpy as np
from numpy import *
import random

def file2matrix(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return mat(dataSet)

def create(n,path):    #随机生成n个二维数据点并保存到路径path中
    file1 = open(path, 'w')
    file1.write("")
    file1.close()
    file = open(path, 'a+')
    for i in range(n):
        a = random.uniform(0.1, 10.0)
        b = random.uniform(0.1, 10.0)
        file.write(str(a) + '\t' + str(b) + '\n')

    file.close()

def cosdistance(A,B):  #余弦距离函数
    result = dot(array(A)[0],array(B)[0])/(linalg.norm(A) * linalg.norm(B))
    return result

def ecluddistance(A,B):  #欧拉距离函数
    result = linalg.norm(A - B)
    return result

def randCenters(dataSet,k):  #在训练集dataSet中随机生成k个初始中心点
    n = shape(dataSet)[1]
    centers =mat(zeros((k,n)))
    for col in range(n):   #循环生成k个
        mincol = min(dataSet[:,col])
        maxcol = max(dataSet[:,col])
        for i in range(k):
            centers[i,col] = mincol + float(maxcol - mincol) * random.uniform(0.0,1.0)
    return centers

def color_clusters(dataindex,dataSet,plt,k=4):
    index = 0
    datalen = len(dataindex)
    for indx in range(datalen):
        if int(dataindex[indx]) == 0:
            plt.scatter(dataSet[index,0],dataSet[index,1],c='blue',marker='o')
        elif int(dataindex[indx]) == 1:
            plt.scatter(dataSet[index, 0], dataSet[index, 1], c='green', marker='o')
        elif int(dataindex[indx]) == 2:
            plt.scatter(dataSet[index, 0], dataSet[index, 1], c='red', marker='o')
        elif int(dataindex[indx]) == 3:
            plt.scatter(dataSet[index, 0], dataSet[index, 1], c='cyan', marker='o')
        index += 1

def drawScatter(plt,mydata,size=20,color='blue',mrkr='o'):
    plt.scatter(mydata.T[0].tolist(),mydata.T[1].tolist(),s=size,c=color,marker=mrkr)


def showplt(plt,centers,ClustDist,dataMat):
    plt.figure()
    color_clusters(ClustDist[:, 0:1], dataMat, plt)
    drawScatter(plt, centers, size=60, color='black', mrkr='D')

