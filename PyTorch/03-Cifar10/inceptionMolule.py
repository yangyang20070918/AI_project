import torch
import torch.nn as nn
import torch.nn.functional as F


'''
input: A

resnet: B = g(A) + f(a)

Inception:
B1 = f1(A)
B2 = f2(A)
B3 = f3(A)

concat([B1, B2, B3])
'''

#定义一个函数将算子串联
def ConvBNRelu(in_channel, out_channel, kernel_size):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=kernel_size, stride=1,
                      padding=kernel_size//2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )


#定义一个类(结构单元)，由多个算子组合成的一个算子
class BaseInception(nn.Module):
    def __init__(self, in_channel, out_channel_list, reduce_channel_list):
        super(BaseInception, self).__init__()
        
        self.branch1_conv = ConvBNRelu(in_channel, out_channel_list[0], 1)

        self.branch2_conv1 = ConvBNRelu(in_channel, reduce_channel_list[0], 1)
        self.branch2_conv2 = ConvBNRelu(reduce_channel_list[0], out_channel_list[1], 3)

        self.branch3_conv1 = ConvBNRelu(in_channel, reduce_channel_list[1], 1)
        self.branch3_conv2 = ConvBNRelu(reduce_channel_list[1], out_channel_list[2], 5)

        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = ConvBNRelu(in_channel, out_channel_list[3], 3)

    
    def forward(self, x):
        out1 = self.branch1_conv(x)
        
        out2 = self.branch2_conv1(x)
        out2 = self.branch2_conv2(out2)

        out3 = self.branch3_conv1(x)
        out3 = self.branch3_conv2(out3)

        out4 = self.branch4_pool(x)
        out4 = self.branch4_conv(out4)

        #进行串联
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        return out


class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        #定义第一组卷积
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BaseInception(in_channel=128, out_channel_list=[64, 64, 64, 64],
                          reduce_channel_list=[16, 16]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block4 = nn.Sequential(
            BaseInception(in_channel=256, out_channel_list=[96, 96, 96, 96],
                          reduce_channel_list=[32, 32]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = F.avg_pool2d(out, 2)
        #fc层之前先展平
        out = out.view(out.size(0), -1)
        #fc层
        out = self.fc(out)
        #out = F.log_softmax(out, dim=1)
        
        return out

def InceptionNetSmall():
    return InceptionNet()
