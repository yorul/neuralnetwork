#!/usr/bin/env python
# coding: utf-8

# In[12]:


import sys,os
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
from two_layer_net import TwoLayerNet

import matplotlib.pyplot as plt

# データの読み込み
test = "test75_iris.dat"
train = "Traning75_iris.dat"

def file_input(path):
    #0で初期化した対応の配列を作成する
    fvalue = np.zeros((75,4)) #４つの特徴量
    speciesn = np.zeros((75,3)) #３つの品種名
    col = 0
    with open(path) as f:
        for datal in f:
            if datal == '\n':
                break
            data = datal.split(',')
            for i in range(0,4):
                fvalue[col][i] = float(data[i])
                '''
                [setosa]      --->  [1 0 0]
                [versicolor]  --->  [0 1 0]
                [virginica]   --->  [0 0 1]
                '''
            if data[4] == 'Iris-setosa\n':
                speciesn[col][0] = 1
            elif data[4] == 'Iris-versicolor\n':
                speciesn[col][1] = 1
            else:
                speciesn[col][2] = 1
            col+=1
        f.close()
    return fvalue, speciesn

(x_train, t_train) = file_input(train)
(x_test, t_test) = file_input(test)

network = TwoLayerNet(input_size=4,hidden_size=50,output_size=3)

itres_num = 1000 #繰り返す回数
train_size = x_train.shape[0]
batch_size = 100 #ミニバッチのサイズ
learning_rate = 0.1

train_loss_list=[]
test_loss_list=[]
train_acc_list=[]
test_acc_list=[]

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(itres_num):
    
    # ミニバッチの取得
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    # 誤差逆伝播法によって勾配を求める
    grad = network.gradient(x_batch,t_batch)

    # パラメータ更新
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    train_loss = network.loss(x_train,t_train)
    train_loss_list.append(train_loss)
    test_loss = network.loss(x_test, t_test)
    test_loss_list.append(test_loss)

    # 1エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train) #訓練データの認識精度
        test_acc = network.accuracy(x_test,t_test) #テストデータの認識精度
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)

#データ（認識精度・損失関数）推移の可視化：グラフをプロット
fig, axs = plt.subplots(4, figsize=(15.0, 15.0))
axs[0].plot(train_acc_list)
axs[0].set_title("train data accuracy")
axs[1].plot(test_acc_list)
axs[1].set_title("test data accuracy")
axs[2].plot(train_loss_list)
axs[2].set_title("train data loss")
axs[3].plot(test_loss_list)
axs[3].set_title("test data loss")

fig.tight_layout()
plt.show()


# In[ ]:




