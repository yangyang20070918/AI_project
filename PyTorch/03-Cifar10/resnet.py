import torch
import torch.nn as nn
import torch.nn.functional as F


#搭建ResNet的核心组件，之后将这组件串联
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1): #初始化
        super(ResBlock, self).__init__()
        #定义主干分支
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        #定义跳跃分支
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)
        return out
        

class ResNet(nn.Module):
    #循环方式写每一层
    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i ==0:
                stride = stride
            else:
                stride = 1
            layers_list.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel

        return nn.Sequential(*layers_list)

    def __init__(self, ResBlock): #初始化
        super(ResNet, self).__init__()
        self.in_channel = 32
        #定义第一组卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        #一层一层写麻烦
##        self.layer1 = ResBlock(in_channel=32, out_channel=64, stride=2)
##        self.layer2 = ResBlock(in_channel=64, out_channel=64, stride=1)
##        self.layer3 = ResBlock(in_channel=64, out_channel=128, stride=2)
        #用循环方式写
        self.layer1 = \
                    self.make_layer(ResBlock, 64, 2, 2)
        self.layer2 = \
                    self.make_layer(ResBlock, 128, 2, 2)
        self.layer3 = \
                    self.make_layer(ResBlock, 256, 2, 2)
        self.layer4 = \
                    self.make_layer(ResBlock, 512, 2, 2)
        #全连接层
        self.fc = nn.Linear(512, 10)

    def forward(self, x): #向前推进
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        #fc层之前先展平
        out = out.view(out.size(0), -1)#-1是自动识别
        #fc层
        out = self.fc(out)
        #out = F.log_softmax(out, dim=1)
        
        return out

def resnet():
    return ResNet(ResBlock)



