import torchvision.transforms as transforms
from torch.utils.data import DataLoader  # 修正：from...import语法
#from PIL import Image
import torch
from models import Discriminator, Generator
#from utils import ReplayBuffer, LambdaLR, weights_init_normal
from datasets import ImageDataset
#import itertools
#import tensorboardX
import os
from torchvision.utils import save_image

#实现从苹果到桔子之间的相互转换
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #network定义网络
    netG_A2B = Generator().to(device) #第一个生成器
    netG_B2A = Generator().to(device) #第二个生成器

    netG_A2B.load_state_dict(torch.load("models/netG_A2B.pth"))
    netG_B2A.load_state_dict(torch.load("models/netG_B2A.pth"))

    #定义当前的状态
    netG_A2B.eval()
    netG_B2A.eval()

    size = 256
    #定义输入
    input_A = torch.ones([1, 3, size, size], 
                         dtype=torch.float).to(device) #初始化
    input_B = torch.ones([1, 3, size, size], 
                         dtype=torch.float).to(device)
    #定义预处理，transform数据增强
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    #传入数据
    #data_root = "datasets/apple2orange"
    data_root = "C:/Users/youyo/Desktop/AI_data/cyclegan/apple2orange"
    
    dataloader = DataLoader(ImageDataset(data_root, transforms_,"test"), 
                            batch_size=1, shuffle=False, num_workers=8)  # 修正：num_workers拼写

    #测试并保存结果
    if not os.path.exists("outputs/A"):
        os.makedirs("outputs/A")
    if not os.path.exists("outputs/B"):
        os.makedirs("outputs/B")

    for i, batch in enumerate(dataloader):
        #不需要input_A.copy_操作
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        #real_A = torch.tensor(input_A.copy_(batch)['A'], dtype=torch.float).to(device)
        #real_B = torch.tensor(input_B.copy_(batch)['B'], dtype=torch.float).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        #保存
        save_image(fake_A, "outputs/A/{}.png".format(i))
        save_image(fake_B, "outputs/B/{}.png".format(i))

        print(i)
#---------------------------------------
if __name__ == '__main__':
    test()
