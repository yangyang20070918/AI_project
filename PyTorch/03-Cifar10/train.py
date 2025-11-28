import torch
import torch.nn as nn
import torchvision
#---------------------------------------------------
#from vggnet import VGGNet #卷积方法1
#from resnet import resnet #卷积方法2
#from mobilenetvl import mobilenetvl_small #卷积方法3
#from inceptionMolule import InceptionNetSmall #卷积方法4
from pre_resnet import pytorch_resnet18 #方法5:使用提供好的模型，将该模定义到pre_resnet供使用
#---------------------------------------------------
from load_cifar10 import train_loader, test_loader
import os
import tensorboardX


if __name__ == '__main__':  #用Windows时，num_workers>0的必须対策要使用__name__ == '__main__'
    #有GPU就在GPU，没有就在CPU上的定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_num = 200 #对样本遍历200次
    lr = 0.01 #学习率
    batch_size = 6 #batch_size的定义
    #---------------------------------------------------
    #net = VGGNet().to(device) #卷积方法1
    #net = resnet().to(device) #卷积方法2
    #net = mobilenetvl_small().to(device) #卷积方法3
    #net = InceptionNetSmall().to(device) #卷积方法4
    net = pytorch_resnet18().to(device) #方法5:使用提供好的模型
    #---------------------------------------------------

    #loss
    loss_func = nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) #优化器1
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) #优化器2

    #调整学习率，指数衰减的方式
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    #step_size=5 没进行5个epoch（5轮遍历）衰减一次，gamma是衰减系数
    #---------------------------------------------------
    #记录,用tensorboardX记录变化过程，以便后续调整模型
    model_path = "models/InceptionNetSmall" #方法4保存
    log_path = "logs/InceptionNetSmall" ##方法4保存

    model_path = "models/pytorch_resnet18" #方法5保存
    log_path = "logs/pytorch_resnet18" ##方法5保存
    #---------------------------------------------------
    
    if not os.path.exists(log_path):
        #os.mkdir(log_path)
        os.makedirs(log_path, exist_ok=True)  # 创建多级目录
    if not os.path.exists(model_path):
        #os.mkdir(model_path)
        os.makedirs(model_path, exist_ok=True)
    writer = tensorboardX.SummaryWriter(log_path)
    
    #----------------
    #学习的过程
    #----------------
    step_n = 0 #定义整体学习的过程,从0开始
    for epoch in range(epoch_num):#epoch 每轮学习
        #print("epoch is ", epoch)
        #----------------
        #每轮的训练
        #----------------
        net.train() #train BN dropout

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad() #优化器梯度为0
            loss.backward() #反向传播
            optimizer.step() #更新参数

            #第一个维度上最大值
            _, pred = torch.max(outputs.data, dim=1)
            #判断拿到的结果是否和label类别标签相同,获得正确的数量
            correct = pred.eq(labels.data).cpu().sum()

##            print("epoch is ", epoch)
##            print("train step", i, "loss is:", loss.item(),
##                 "mini-batch correct is:", 100.0 * correct / batch_size)
##            print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"]) #打印学习率

            #用tensorboardX记录变化过程，以便后续调整模型
            writer.add_scalar("train loss", loss.item(), global_step=step_n)
            writer.add_scalar("train correct", 100.0 * correct.item() / batch_size, global_step=step_n)
            #记录里加入图片,以便查看
            im = torchvision.utils.make_grid(inputs)
            writer.add_image("train im", im, global_step=step_n)
            step_n += 1
            
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(net.state_dict(), "models/{}.pth".format(epoch + 1))
        
        scheduler.step() #对学习率进行更新
        print("train epoch is ", epoch, "lr is ", optimizer.state_dict()["param_groups"][0]["lr"]) #打印学习率

        #----------------
        #每轮训练后进行测试
        #----------------
        #测试(只做前向运算，不做反向传播)
        sum_loss = 0
        sum_correct = 0
        net.eval() #做测试用
        with torch.no_grad():  # 添加这行
            for i,data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = loss_func(outputs, labels)
                #第一个维度上最大值
                _, pred = torch.max(outputs.data, dim=1)
                #判断拿到的结果是否和label类别标签相同,获得正确的数量
                correct = pred.eq(labels.data).cpu().sum()

                sum_loss += loss.item()
                sum_correct += correct.item()

        test_loss = sum_loss * 1.0 / len(test_loader)
        test_correct = sum_correct * 100.0 / len(test_loader) / batch_size

        print("epoch is",epoch + 1, "loss is:", test_loss,
              "test correct is:", test_correct)

        #用tensorboardX记录变化过程，以便后续调整模型
        writer.add_scalar("test loss", test_loss, global_step=epoch + 1) #记录每个epoch对应的loss和准确率
        writer.add_scalar("test correct", test_correct, global_step=epoch + 1)
        #记录里加入图片,以便查看
        im = torchvision.utils.make_grid(inputs)
        writer.add_image("test im", im, global_step=step_n)
        
        

        
