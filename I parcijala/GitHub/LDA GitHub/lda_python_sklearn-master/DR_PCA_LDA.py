#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Python 实现PCA以及LDA算法
#Author：Rui Wang
#Date: 2017.07.09
#加载库
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
import scipy
import matplotlib.pyplot as plt

#加载数据
data = load_iris()
x = data['data']
y = data['target']
x_new = x.T #4*150,每一列是一个样本，共有150列也即150个样本，同时他们属于三个类
#对数据进行标准化处理
x_s = scale(x_new, with_mean = True, with_std = True, axis = 1)#按行求均值

#构造协方差矩阵以及进行特征分解
x_cov = np.cov(x_s) #该函数内部在处理数据时会先对数据进行标准化的corrcoef

#特征分解
eig_val, eig_vec = scipy.linalg.eig(x_cov)
print ("Eigen values \n%s"%(eig_val))
print ("\n Eigen vectors \n%s"%(eig_vec))
print ("\n所求特征向量为：")
eig_val = eig_val.real #取复数的实部
sum_eig_val = np.sum(eig_val)#特征值的总和5
each_val_percent = 0
sum_each_val_percent = 0
count = 0
#根据能量大小确定所选的特征向量的个数
for i,each_val in enumerate(eig_val):
    each_val_percent = round((each_val / sum_eig_val),3)
    sum_each_val_percent += each_val_percent
    if (sum_each_val_percent < 0.997):
        count += 1
aim_W = eig_vec[:,0:count] #此处的特征值已按大小排好序

#对原始数据进行降维
x_new_dr = aim_W.T.dot(x_new)
#画出二维散点图, 观察PCA操作区分不同类别样本的能力
plt.figure(1)
plt.scatter(x_new_dr[0,:],x_new_dr[1,:],c=y)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
#对降维后的样本进一步实施鉴别分析也即LDA
#第一步是找到同类样本
class_label = [i for i in range(3)] #类别标签，共有0,1,2三类
temp = []
#temp = list()
#Sequence_sample = list()
Sequence_sample = [[]] #for k in range(3)
for j in class_label:
    x_each_label = np.where(y == j)[0]#找出具有该类别标签的行
    temp = x_new_dr[:,x_each_label]#取出属于同一类的样本赋给中间变量
    Sequence_sample.append(temp)#按类别把样本顺序从新排列

#第二部是计算类内离散度
len_sample = len(Sequence_sample)
mean_class = []

Sw = [[0 for i_w in range(len_sample-1)] for j_w in range(len_sample-1)]
sw = [[0 for i_w in range(len_sample-1)] for j_w in range(len_sample-1)]
Sb = [[0 for i_w in range(len_sample-1)] for j_w in range(len_sample-1)]

for l in range(1,len_sample):
    temp_ =  Sequence_sample[l]
    mean_each_class = np.mean(temp_,axis=1)#按行求均值
    mean_class.append(mean_each_class)

for m in range(1,len_sample):
    _temp_ = Sequence_sample[m]
    for n in range(0,_temp_.shape[1]):
        temp_intra_cov = _temp_[:,n]-mean_class[m-1] #_temp_[:,n]
        temp_intra_cov = np.array([temp_intra_cov])
        sw += np.dot(temp_intra_cov.T,temp_intra_cov)
    Sw += sw

#第三步是计算类间离散度
mean_all_sample = []
mean_all_sample = np.mean(x_new_dr,axis=1)
#mean_all_sample = mean_all_sample.tolist()

for o in range(1,len_sample):
    temp_inter = mean_class[o-1]-mean_all_sample
    temp_inter = np.array([temp_inter])
    _compute = Sequence_sample[o].shape[1]
    #constant_list = [[_compute for p in range(len_sample-1)] for q in range(len_sample-1)]
    temp_inter_cov = _compute * np.dot(temp_inter.T,temp_inter)
    #for r in range(0,len_sample-1):
        #for s in range(len_sample):
            #temp_dot = temp_inter_cov[r][s]*_compute
    Sb += temp_inter_cov

#第四步，计算inv(Sw)*Sb,并进行特征分解
Sw = np.linalg.inv(Sw)#求逆
temp_lda = np.dot(Sw,Sb)
eig_val_lda, eig_vec_lda = scipy.linalg.eig(temp_lda)#特征分解
num_classes= len(np.unique(y))
aiw_W_lda = eig_vec_lda[:,0:num_classes-1]#最终的投影向量

#第五步，对样本进行投影操作
sample_lda = np.dot(aiw_W_lda.T,x_new_dr) #2*150

#第六步，画出新的二维散点图,可以发现通过LDA算法，不同类别之间相对于PCA算法分的更加的开了
plt.figure(2)
plt.scatter(sample_lda[0,:],sample_lda[1,:],c=y)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()