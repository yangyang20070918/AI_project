import torch
import torch.nn as nn
import torch.nn.functional as F


#搭建ResNet的核心组件，之后将这组件串联
class ResBlock(nn.Module):
    def __init__(self, in_channel): #初始化
        super(ResBlock, self).__init__()
        #定义卷积的核心模块
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel,kernel_size=3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel,kernel_size=3),
            nn.InstanceNorm2d(in_channel),
        ]
    
        #
        self.conv_block = nn.Sequential(*conv_block)

    #卷积的跳连
    def forward(self, x):
        return x + self.conv_block(x)
        
#生成器
class Generator(nn.Module):
    def __init__(self): #初始化
        super(Generator, self).__init__()
        self.in_channel = 32
        #定义第一组卷积
        net = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64,kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        #downsample定义下采样模块
        in_channel = 64
        out_channel = in_channel * 2
        for _ in range(2):
            net += [
                nn.Conv2d(in_channel, out_channel,kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel * 2

        #定义ResBlock
        for _ in range(9):
            net += [ResBlock(in_channel)]

        #upsample定义上采样模块,用反卷积方式，刚好和上采样反过来
        out_channel = in_channel // 2
        for _ in range(2):
            net += [
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3,
                                  stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ]
            in_channel = out_channel
            out_channel = in_channel // 2

        #定义输出层
        net += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channel, 3,kernel_size=7),
            nn.Tanh()
            
        ]

        #定义我们的model，将定义好的算子进行串联
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)
        

#判别器
class Discriminator(nn.Module):
    def __init__(self): #初始化
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(3, 64,kernel_size=4,stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128,kernel_size=4,stride=2, padding=1),
                 nn.InstanceNorm2d(128),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256,kernel_size=4,stride=2, padding=1),
                 nn.InstanceNorm2d(256),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512,kernel_size=4,stride=2, padding=1),
                 nn.InstanceNorm2d(512),
                 nn.LeakyReLU(0.2, inplace=True)]
        #最后映射到一维
        model += [nn.Conv2d(512, 1,kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


#对模型进行测试
if __name__=='__main__':
    G = Generator()
    D = Discriminator()
    #定义一个tensor
    input_tensor = torch.ones((1, 3, 256, 256),dtype=torch.float)
    #用生成器对input_tensor进行前向推理
    out = G(input_tensor)
    print(out.size())
    #用判别器对input_tensor进行前向推理
    out = D(input_tensor)
    print(out.size())




