#cifar10数据下载解析

import pickle
#以下是官网给的图片读取的代码，保存为字典
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#类别名称（官网提供）
label_name = ["airplane",
             "automobile",
             "bird",
             "cat",
             "deer",
             "dog",
             "frog",
             "horse",
             "ship",
             "truck"]

#拿数据，解析数据
import glob
import numpy as np
import cv2
import os

#训练用图片
#train_list = glob.glob("C:/Users/youyo/Desktop/AI_学習/PyTorch_項目/03-Cifar10/cifar-10-python/cifar-10-batches-py/data_batch_*")
#train_list = glob.glob("/cifar-10-python/cifar-10-batches-py/data_batch_*")
#train_list = glob.glob("C:/Users/youyo/Desktop/AI_data/cifar-10-python/cifar-10-batches-py/data_batch_*") #路径有中文时保存图片失败

#测试用图片
train_list = glob.glob("C:/Users/youyo/Desktop/AI_data/cifar-10-python/cifar-10-batches-py/test_batch*") 

print(train_list)

#存放图片的位置（读取到图片后需要存放）
#训练用图片
#save_path = "C:/Users/youyo/Desktop/AI_学習/PyTorch_項目/03-Cifar10/cifar-10-python/cifar-10-batches-py/TRAIN"
#save_path = "/cifar-10-python/cifar-10-batches-py/TRAIN"
save_path = "C:/Users/youyo/Desktop/AI_data/cifar-10-python/cifar-10-batches-py/TRAIN" #路径有中文时保存图片失败

#测试用图片
save_path = "C:/Users/youyo/Desktop/AI_data/cifar-10-python/cifar-10-batches-py/TEST" 

for l in train_list:
    print(l)
    l_dict = unpickle(l)
    #print(l_dict)
    print(l_dict.keys())
    #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    #batch_label :第几个文件（这里一共有5个文件）
    #labels :类别id号(这里一共10个类别，例如第一个类别的id为0)
    #data :3*32*32维度的向量
    #filenames :图片的名字

    for im_idx, im_data in enumerate(l_dict[b'data']):
        #print(im_idx)
        #print(im_data)
        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]
        #print (im_idx,im_label,im_name,im_data)
        #9999 1 b'coupe_s_001616.png' [229 236 234 ... 173 162 161]

        im_label_name = label_name[im_label]#获得类别的名字
        im_data = np.reshape(im_data, [3, 32, 32])#将数据reshape维度3乘32乘32
        im_data = np.transpose(im_data, (1, 2, 0))#将通道channel的位置变换，将第0维放到最后面(32，32，3的顺序)
        
        #可视化
        #cv2.imshow("im_data",im_data)#将图片可视化
        #cv2.imshow("im_data",cv2.resize(im_data, (200, 200)))#cv2.resize调整图片大小
        #cv2.waitKey(0)#加这行代码的作用是，安 空格时图片才会跳转至下一张

        #类别的文件夹没有的话，创建文件夹
        if not os.path.exists("{}/{}".format(save_path, im_label_name)):
            os.makedirs("{}/{}".format(save_path, im_label_name))
    
        #写入图片
        cv2.imwrite("{}/{}/{}".format(save_path, 
                                      im_label_name, 
                                      im_name.decode("utf-8")), #图片的名字是byte型这里需要转换，解码成utf-8的格式
                   im_data)
        
        
