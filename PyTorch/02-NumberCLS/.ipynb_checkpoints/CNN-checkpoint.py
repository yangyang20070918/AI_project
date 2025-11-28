import torch
#（2）net 搭建网络模型
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #卷积层定义
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2), #图片是灰度图channel为1，输出channel定义为32，卷积核定义为5
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        #定义线性操作，将卷积之后的结果输入给他
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)
        #参数要思考一下，输入的图片是28*28,第一轮卷积后因为MaxPool2d(2)所以大小变为14*14，
        #channel是32，
        #输出为10，因为一共有0-9共10个数字，所以为10维

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        #--------
        #out = self.predict(x)
        return(out)
