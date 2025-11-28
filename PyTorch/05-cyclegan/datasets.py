#数据读取
import glob #用于读取文件夹下的文件
import random
from torch.utils.data import Dataset #数据读取
from PIL import Image #图片读取的读取工具
import torchvision.transforms as trForms #数据增强
import os


#定义类
class ImageDataset(Dataset):
    def __init__(self, root="", transform=None, model="train"): #初始化函数，完成数据的读取
        #定义变量
        self.transform = trForms.Compose(transform)

        #self.pathA = os.path.join(root, model, "A/*")
        #self.pathB = os.path.join(root, model, "B/*")
        self.pathA = os.path.join(root, f"{model}A", "*")
        self.pathB = os.path.join(root, f"{model}B", "*")
        
        self.list_A = glob.glob(self.pathA)
        self.list_B = glob.glob(self.pathB)
        
    def __getitem__(self, index): #用来读取图片样本,增强数据
        im_pathA = self.list_A[index % len(self.list_A)]
        im_pathB = random.choice(self.list_B)

        im_A = Image.open(im_pathA)
        im_B = Image.open(im_pathB)
        
        #数据增强
        item_A = self.transform(im_A)
        item_B = self.transform(im_B)

        return {"A":item_A, "B":item_B}
        
    def __len__(self): #返回样本数量
        return max(len(self.list_A), len(self.list_B))


if __name__=='__main__':
    from torch.utils.data import DataLoader

    root = "C:/Users/youyo/Desktop/AI_data/cyclegan/apple2orange"

    transform_ = [trForms.Resize(256, Image.BILINEAR), trForms.ToTensor()]
    dataloader = DataLoader(ImageDataset(root, transform_, "train"),
                            batch_size=1, shuffle=True, num_workers=1)

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch)




