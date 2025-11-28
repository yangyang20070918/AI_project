#手写数据识别（分类问题）
import torch
import torchvision.datasets as dataset #这个包定义了很多公开的数据集
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN

#（1）data 数据
#手写数字的数据集在数据包中存在，可以直接利用
test_data = dataset.MNIST(root="mnisst", #测试集 保存路径
                          train=False, #数据一共有6万涨训练集和1万张测试集False是选择测试集
                          transform=transforms.ToTensor(), #将数据集保存为tensor
                          download=False)
#--------
#！！数据过大时，内存有限无法直接用numpy全部读取，所以这里需要分小块读取！！
#--------
#batchsize
test_loader = data_utils.DataLoader(dataset= test_data, #数据分批提取
                                     batch_size=64, #一般为64
                                     shuffle=True) 

#--------
#（8）load 加载模型
#--------
cnn = torch.load("model/mnist_model.pkl") 
#cnn = cnn.cuda()

#（6）test/eval 测试
loss_test = 0
accuracy = 0
for i, (images, labels) in enumerate(test_loader):
    #images = images.cuda() #使用GPU
    #labels = labels.cuda() #使用GPU
    outputs = cnn(images)
    _, pred = outputs.max(1)
    accuracy = (pred == labels).sum().item() #获取正确的样本数量

accuracy = accuracy / len(test_data)
print(accuracy)


#（9）infercence

