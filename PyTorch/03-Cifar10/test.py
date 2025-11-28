import torch
import glob
import cv2
from PIL import Image #图片读取的读取工具
from torchvision import transforms #数据增强
import numpy as np #数据以np形式存储
from inceptionMolule import InceptionNetSmall


if __name__ == '__main__':  #用Windows时，num_workers>0的必须対策要使用__name__ == '__main__'
    #有GPU就在GPU，没有就在CPU上的定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = InceptionNetSmall()

    net.load_state_dict(torch.load("C:/Users/youyo/Desktop/AI_data/cifar-10-python/models/200.pth"))
    #图片
    im_list = glob.glob("C:/Users/youyo/Desktop/AI_data/cifar-10-python/cifar-10-batches-py/TEST/*/*.png")

    np.random.shuffle(im_list)

    net.to(device) #放到GPU
    #标签
    label_name = ["airplane","automobile","bird","cat","deer",
             "dog","frog","horse","ship","truck"]

    #用来对测试数据的增强
    test_transform = transforms.Compose([
        transforms.CenterCrop((28,28)),
        transforms.ToTensor(), #转化为网络输出的数据
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)), 
    ])

    #遍历图片数据
    for im_path in im_list:
        net.eval()
        im_data = Image.open(im_path)

        inputs = test_transform(im_data)
        inputs = torch.unsqueeze(inputs, dim=0) #加入一个维度
        inputs = inputs.to(device) #GPU
        
        outputs = net.forward(inputs)
        #将输出结果转换
        _, pred = torch.max(outputs.data, dim=1)
        print(label_name[pred.cpu().numpy()[0]])

        #转换格式数据
        img = np.asarray(im_data)
        #img = img[:, :, [1, 2, 0]] #可能是错的，改为[2, 1, 0],直接用下面的方法也行
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #修正颜色转换,# PIL读取是RGB，OpenCV需要BGR

        img = cv2.resize(img, (300, 300))
        cv2.imshow("im", img)
        cv2.waitKey()

        
