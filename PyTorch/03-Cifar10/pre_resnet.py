#pytorch 有提供好的模型可以直接用,不需要自定义网络结构
import torch
import torch.nn as nn
#from torchvision import models #该方法已废弃
from torchvision.models import ResNet18_Weights


#使用models定义好的模型，这里以resnet18为例
class resnet18(nn.Module):
    def __init__(self): #初始化
        super(resnet18, self).__init__()
        #定义models.resnet18模型（#还有很多其他模型可以使用）
        #self.model = models.resnet18(pretrained=True) #如果没有这个模型的话会自动下载
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) #上面的参数已废弃，用这个新参数定义
        self.num_features = self.model.fc.in_features #特征向量的一个维度
        self.model.fc = nn.Linear(self.num_features, 10) #本次使用自例子是10分类，所以为10
        

    def forward(self, x):
        out = self.model(x)
        return out

def pytorch_resnet18():
    return resnet18()
        




