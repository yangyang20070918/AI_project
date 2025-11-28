import torchvision.transforms as transforms
from torch.utils.data import DataLoader  # 修正：from...import语法
from PIL import Image
import torch
from models import Discriminator, Generator
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from datasets import ImageDataset
import itertools
import tensorboardX



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    size = 256
    lr = 0.0002 #初始学习率
    n_epoch = 4#200 #训练次数
    epoch = 0
    decay_epoch = 2#100 #衰减的训练

    #network定义网络
    netG_A2B = Generator().to(device) #第一个生成器
    netG_B2A = Generator().to(device) #第二个生成器
    netD_A = Discriminator().to(device) #第一个判别器
    netD_B = Discriminator().to(device) #第二个判别器

    #定义loss
    loss_GAN = torch.nn.MSELoss()
    loss_cycle = torch.nn.L1Loss()
    loss_identity = torch.nn.L1Loss() #比较相似程度

    #定义优化器
    opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),  # 修正：parameters()拼写
                            lr=lr, betas=(0.5, 0.9999)) #将两个网络的参数进行连接

    opt_DA = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.9999))  # 修正：parameters()拼写
    opt_DB = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.9999))  # 修正：parameters()拼写

    #定义学习率衰减的方法
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)

    lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
    lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)

    #定义训练数据
    data_root = "C:/Users/youyo/Desktop/AI_data/cyclegan/apple2orange"
    input_A = torch.ones([1, 3, size, size], 
                        dtype=torch.float).to(device) #初始化
    input_B = torch.ones([1, 3, size, size], 
                        dtype=torch.float).to(device) #初始化
    #标签真样本
    label_real = torch.ones([1, 1], dtype=torch.float, 
                            requires_grad=False).to(device)
    #标签假样本
    label_fake = torch.zeros([1, 1], dtype=torch.float, 
                            requires_grad=False).to(device)

    #定义两个buff
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    #定义log路径
    log_path = "logs"
    #写log的读写器
    writer_log = tensorboardX.SummaryWriter(log_path)

    #定义transform数据增强
    transforms_ = [
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    #定义对数据进行加载的dataloader
    dataloader = DataLoader(ImageDataset(data_root, transforms_), 
                            batch_size=batch_size, shuffle=True, num_workers=8)  # 修正：num_workers拼写

    step = 0 #计数器

    for epoch in range(n_epoch):
        print("epoch=", epoch)
        for i, batch in enumerate(dataloader):
            # 修正：直接从batch中获取数据，不需要input_A.copy_操作
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            opt_G.zero_grad()

            same_B = netG_A2B(real_B)
            loss_identity_B = loss_identity(same_B, real_B) * 5.0

            same_A = netG_B2A(real_A)
            loss_identity_A = loss_identity(same_A, real_A) * 5.0
            
            
            fake_B = netG_A2B(real_A) #用A生成假的B
            pred_fake = netD_B(fake_B) #用B的判别器判断假B的真假
            loss_GAN_A2B = loss_GAN(pred_fake, label_real)

            fake_A = netG_B2A(real_B) #用B生成假的A
            pred_fake = netD_A(fake_A) #用A的判别器判断假A的真假
            loss_GAN_B2A = loss_GAN(pred_fake, label_real)

            #cycle loss
            recovered_A = netG_B2A(fake_B) #用假的B恢复A
            loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0 #用恢复的A与真的A进行一致性判断

            recovered_B = netG_A2B(fake_A) #用假的A恢复B
            loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0 #用恢复的B与真的B进行一致性判断
            #total loss
            # 修正：最后应该是loss_cycle_BAB而不是loss_GAN_A2B
            loss_G = loss_identity_A + loss_identity_B + \
                    loss_GAN_A2B + loss_GAN_B2A + \
                    loss_cycle_ABA + loss_cycle_BAB
            
            
            loss_G.backward() #对生成器进行反向传播
            opt_G.step()
            
            #----------------判别器----------------
            #----对判别器A
            opt_DA.zero_grad()
            pred_real = netD_A(real_A) #对判别器进行预测
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = loss_GAN(pred_fake, label_fake)  # 修正：应该是pred_fake和label_fake

            #total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5  # 修正：应该是乘以0.5
            loss_D_A.backward() #对判别器进行反向传播
            opt_DA.step()

            #----对判别器B
            opt_DB.zero_grad()
            pred_real = netD_B(real_B) #对判别器进行预测
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = loss_GAN(pred_fake, label_fake)  # 修正：应该是pred_fake和label_fake

            #total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5  # 修正：应该是乘以0.5
            loss_D_B.backward() #对判别器进行反向传播
            opt_DB.step()

            print("loss_G:{}, loss_G_identity:{}, loss_G_GAN:{}, loss_G_cycle:{}, loss_D_A:{}, loss_D_B:{}".format(
                loss_G, loss_identity_A + loss_identity_B,
                loss_GAN_A2B + loss_GAN_B2A,
                loss_cycle_ABA + loss_cycle_BAB,
                loss_D_A, loss_D_B
            ))

            writer_log.add_scalar("loss_G", loss_G, global_step=step+1)
            writer_log.add_scalar("loss_G_identity", loss_identity_A + loss_identity_B, global_step=step+1)
            writer_log.add_scalar("loss_G_GAN", loss_GAN_A2B + loss_GAN_B2A, global_step=step+1)
            writer_log.add_scalar("loss_G_cycle", loss_cycle_ABA + loss_cycle_BAB, global_step=step+1)
            writer_log.add_scalar("loss_D_A", loss_D_A, global_step=step+1)
            writer_log.add_scalar("loss_D_B", loss_D_B, global_step=step+1)

            step += 1

        #更新学习率
        lr_scheduler_G.step()
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()
        #每训练完一个，保存4个模型，两个生成器，两个判别器
        torch.save(netG_A2B.state_dict(), "models/netG_A2B.pth")
        torch.save(netG_B2A.state_dict(), "models/netG_B2A.pth")
        torch.save(netD_A.state_dict(), "models/netD_A.pth")
        torch.save(netD_B.state_dict(), "models/netD_B.pth")
