import torch
import numpy as np
import re

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

#（1）data 数据
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

#已做好的模型进行读取
net = torch.load("model/model.pkl", weights_only=False)#读取模型
#weights_only=False 是关闭安全检查，检查模型是否信任的问题，这里不需要检查所以关闭

#(3)loss 损失函数
loss_func = torch.nn.MSELoss()

#(6)test 测试
x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32) #定义样本标签
pred = net.forward(x_data) #定义网络的向前运算
pred = torch.squeeze(pred)#为了让维度一致这里将pred降维
loss_test = loss_func(pred, y_data) * 0.001 
print("loss_test{}".format(loss_test))#打印迭代次数和loss的值


