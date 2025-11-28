#数据读取
from torchvision import transforms #数据增强
from torch.utils.data import DataLoader, Dataset #数据加载和数据读取
import os
from PIL import Image #图片读取的读取工具
import numpy as np #数据以np形式存储
import glob

#定义类别一共10个
label_name = ["airplane","automobile","bird","cat","deer",
             "dog","frog","horse","ship","truck"]

label_dict = {} #用来将标签对应到0-9的数字。
for idx, name in enumerate(label_name):
    label_dict[name] = idx

#用于对图像数据的读取
def default_loader(path):
    return Image.open(path).convert("RGB")

#用来对训练数据的增强
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop((28,28)),
#     transforms.RandomHorizontalFlip(), #随机垂直翻转
#     transforms.RandomVerticalFlip(), #随机水平翻转
#     transforms.RandomRotation(90), #角度会在-90与90之间进行旋转
#     transforms.RandomGrayscale(0.1), #随机转化灰度
#     transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), #颜色的增强
#     transforms.ToTensor() #转化为网络输出的数据
# ])

train_transform = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(), #随机垂直翻转
    transforms.ToTensor() #转化为网络输出的数据
])
#用来对测试数据的增强
test_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor() #转化为网络输出的数据
])


#定义核心类
class MyDataset(Dataset):
    def __init__(self, im_list, transform=None, loader=default_loader): #初始化函数，完成数据的读取
        #im_list 图片的列表,每一张图片的路径
        #transform=None 用于数据增强的函数
        #loader=default_loader #用于对图像数据的读取
        super(MyDataset, self).__init__()
        imgs = []
        for im_item in im_list:
            #"C:/Users/youyo/Desktop/AI_data/cifar-10-python/cifar-10-batches-py/TRAIN/" \
            #"airplane/aeroplane_s_000004.png"
            #因为路径的最后是图片名字，倒数第二项是类别名，所以这里把路径的数据切割取得名字
            #im_label_name = im_item.split("/")[-2] #类别名,分割结果不稳定，前面/后面\
            im_label_name = os.path.normpath(im_item).split(os.sep)[-2] #类别名,这种方法稳定
            imgs.append([im_item, label_dict[im_label_name]]) #存储图片的路径和类别id
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
    def __getitem__(self, index): #用来读取图片样本,增强数据
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path) #读取图片
        #数据增强
        if self.transform is not None:
            im_data = self.transform(im_data)
        return im_data, im_label
    def __len__(self): #返回样本数量
        return len(self.imgs)

#图片列表
im_train_list = glob.glob("C:/Users/youyo/Desktop/AI_data/cifar-10-python/cifar-10-batches-py/TRAIN/*/*.png")
# print("找到的训练图片数量:", len(im_train_list))
# print("前 3 个路径示例:")
# for i in range(min(3, len(im_train_list))):
#     print(im_train_list[i])

im_test_list = glob.glob("C:/Users/youyo/Desktop/AI_data/cifar-10-python/cifar-10-batches-py/TEST/*/*.png")
#读取数据
train_dataset = MyDataset(im_train_list,
                        transform = train_transform) #训练数据做增强处理

test_dataset = MyDataset(im_test_list,transform = test_transform) #测试数据不做增强处理

train_loader = DataLoader(dataset=train_dataset, 
                               batch_size=6, 
                               shuffle=True,
                               num_workers=0) #用4个worker对数据进行加载,windows时为0

test_loader = DataLoader(dataset=test_dataset, 
                               batch_size=6, 
                               shuffle=False,
                               num_workers=0) #用4个worker对数据进行加载,windows时为0

print ("num_of_train", len(train_dataset))
print ("num_of_test", len(test_dataset))
