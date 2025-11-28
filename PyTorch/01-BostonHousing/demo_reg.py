#data 数据
#net 搭建网络模型
#loss 损失函数
#optimiter 优化方法
#training 训练
#test/eval 测试
#save 保存模型
#load 加载模型
#infercence

import torch
#（1）data 数据
import numpy as np
import re
ff = open("housing.data").readlines()
data = []
for item in ff:
    out = re.sub(r"\s{2,}"," ", item).strip() #将多个空格转换成一个空格
    #print (out)
    data.append(out.split(" ")) #将数据以空格分割
data = np.array(data).astype(float)
print (data.shape) #拿到数据

Y = data[:, -1] #对数据进行切分
X = data[:, 0:-1]

X_train = X[0:496, ...]#训练集
Y_train = Y[0:496, ...]
X_test = X[496:, ...]#测试集
Y_test = Y[496:, ...]

print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)

#(2)net 搭建网络模型
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        #--------为了结果的精确度可以让模型变得复杂一点，加入一个隐藏层
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)
        #--------
        #self.predict = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):#一维，线性层的神经网络
        #--------加入隐藏层后这里要调入隐藏层
        out = self.hidden(x)
        out = torch.relu(out)#加入relu非线性运算
        out = self.predict(out)
        #--------
        #out = self.predict(x)
        return(out)

net = Net(13, 1)#输入13个特征，输出1个特征向量

#(3)loss 损失函数
loss_func = torch.nn.MSELoss()

#(4)optimiter 优化(定义学习器)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.0001) #lr定义学习率
#SGD学习器可以换成其他的试试（Adam会比SGD好一点）
#学习率也可以调整
optimizer = torch.optim.Adam(net.parameters(), lr=0.01) #lr定义学习率


#(5)training 训练
for i in range(10000):#训练1000次
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32) #定义样本标签

    pred = net.forward(x_data) #定义网络的向前运算
    #print(pred.shape) #此处是二维
    #print(y_data.shape) #此处是一维
    pred = torch.squeeze(pred)#为了让维度一致这里将pred降维

    #计算loss，需要维度一致的前提下进行计算
    #loss = loss_func(pred, y_data)
    loss = loss_func(pred, y_data) * 0.001 #损失值太大结果为nan出不来，所以要缩小损失的值

    optimizer.zero_grad()#调用优化器，首先参数为0
    loss.backward()#进行反向传播
    optimizer.step()#进行网络更新

    print("ite:{}, loss_train{}".format(i, loss))#打印迭代次数和loss的值
    print(pred[0:10])#预测结果的前十个值
    print(y_data[0:10])#真实结果的前十个值
    

#(6)test 测试
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32) #定义样本标签
    pred = net.forward(x_data) #定义网络的向前运算
    pred = torch.squeeze(pred)#为了让维度一致这里将pred降维
    loss_test = loss_func(pred, y_data) * 0.001 
    print("ite:{}, loss_test{}".format(i, loss_test))#打印迭代次数和loss的值

#保存模型
#训练的loss和测试的loss_test进行比较
torch.save(net, "model/model.pkl") #模型整体保存
#torch.load("")#加载模型
#torch.save(net.state_dict(), "params.pkl") #模型的参数保存（比较小）
#net.load_state_dict("")




