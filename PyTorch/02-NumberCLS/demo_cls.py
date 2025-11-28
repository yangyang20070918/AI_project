#手写数据识别（分类问题）
import torch
import torchvision.datasets as dataset #这个包定义了很多公开的数据集
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN

#（1）data 数据
#手写数字的数据集在数据包中存在，可以直接利用
train_data = dataset.MNIST(root="mnisst", #训练集 保存路径
                          train=True, #数据一共有6万涨训练集和1万张测试集True是选择训练集
                          transform=transforms.ToTensor(), #将数据集保存为tensor
                          download=True) #当前路径没有该数据的话会下载

test_data = dataset.MNIST(root="mnisst", #测试集 保存路径
                          train=False, #数据一共有6万涨训练集和1万张测试集False是选择测试集
                          transform=transforms.ToTensor(), #将数据集保存为tensor
                          download=False)
#--------
#！！数据过大时，内存有限无法直接用numpy全部读取，所以这里需要分小块读取！！
#--------
#batchsize
train_loader = data_utils.DataLoader(dataset= train_data, #数据分批提取
                                     batch_size=64, #一般为64
                                     shuffle=True) 

test_loader = data_utils.DataLoader(dataset= test_data, #数据分批提取
                                     batch_size=64, #一般为64
                                     shuffle=True)



#（2）net 搭建网络模型
cnn = CNN() #初始化这个类
#cnn = cnn.cuda() #转移到GPU上

#（3）loss 损失函数
loss_func = torch.nn.CrossEntropyLoss() #分类问题可以用交叉方

#（4）optimiter 优化方法
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01) #cnn.parameters()为卷积神经网络的参数，lr为学习率

#（5）training 训练
#分类问题里，需要对样本进行重复多轮的训练
for epoch in range(10):#进行10轮,用CPU时选小点
    for i, (images, labels) in enumerate(train_loader):
        #images = images.cuda() #使用GPU
        #labels = labels.cuda() #使用GPU

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward() #通过反向传播来完成优化
        optimizer.step()
        # print("epoch is {}, ite is "
        #      "{}/{}, loss is {}".format(epoch + 1,
        #                                 i,
        #                                 len(train_data) // 64,
        #                                 loss.item()
        #                                ))

    #（6）test/eval 测试
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        #images = images.cuda() #使用GPU
        #labels = labels.cuda() #使用GPU
        outputs = cnn(images)
        #[batchsize]
        #outputs = batchsize * cls_num
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy = (pred == labels).sum().item() #获取正确的样本数量

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)

    print("epoch is {}, accuracy is {},"
          "loss_test is {}".format(epoch + 1,
                                    accuracy,
                                    loss_test.item()
                                   ))

#（7）save 保存模型
torch.save(cnn, "model/mnist_model.pkl")

#（8）load 加载模型

#（9）infercence

