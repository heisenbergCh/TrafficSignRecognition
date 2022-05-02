from __future__ import print_function, division
 
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import json

## 继承nn.Module的Net类
class LeNet(nn.Module):
    '''
    该类继承了torch.nn.Modul类
    构建LeNet神经网络模型
    '''
    def __init__(self):
        super(LeNet, self).__init__()  # 这一个是python中的调用父类LeNet的方法，因为LeNet继承了nn.Module，如果不加这一句，无法使用导入的torch.nn中的方法，这涉及到python的类继承问题，你暂时不用深究

        # 第一层神经网络，包括卷积层、线性激活函数、池化层
        self.conv1 = nn.Sequential(     # input_size=(1*28*28)：输入层图片的输入尺寸，我看了那个文档，发现不需要天，会自动适配维度
            nn.Conv2d(3,6,5),   # padding=2保证输入输出尺寸相同：采用的是两个像素点进行填充，用尺寸为5的卷积核，保证了输入和输出尺寸的相同
            nn.ReLU(),                  # input_size=(6*28*28)：同上，其中的6是卷积后得到的通道个数，或者叫特征个数，进行ReLu激活
            nn.MaxPool2d(kernel_size=2, stride=2), # output_size=(6*14*14)：经过池化层后的输出
        )

        # 第二层神经网络，包括卷积层、线性激活函数、池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # input_size=(6*14*14)：  经过上一层池化层后的输出,作为第二层卷积层的输入，不采用填充方式进行卷积
            nn.ReLU(),            # input_size=(16*10*10)： 对卷积神经网络的输出进行ReLu激活
            nn.MaxPool2d(2, 2)    # output_size=(16*5*5)：  池化层后的输出结果
        )

        # 全连接层(将神经网络的神经元的多维输出转化为一维)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 53 * 53, 120),  # 进行线性变换
            nn.ReLU()                    # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(-1, 16 * 53 * 53)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
plt.ion()   # interactive mode
# 模型存储路径
model_save_path = '/home/wangbaoxing/use_here/traffic/model/net_030.pth'
 
# ------------------------ 加载数据 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定义预训练变换
preprocess_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
 
classes = (0,1,2,3,4,5,6,7,8,9) # 这个顺序很重要，要和训练时候的类名顺序一致
filePath = '/home/wangbaoxing/use_here/traffic/data/test'
filenodelist = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

# ------------------------ 载入模型 --------------------------- #
model = torch.load(model_save_path)
model.eval()
# print(model)

for dirpath,dirnames,filenames in os.walk(filePath):
    for dirname in dirnames:
        for cdirpath,cdirnames,cfilenames in os.walk(filePath+'/'+dirname):
            for filename in cfilenames:
                realName=dirname+'/'+filename

                image_PIL = Image.open(filePath+'/'+realName)
                #
                image_tensor = preprocess_transform(image_PIL)
                # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
                image_tensor.unsqueeze_(0)
                # 没有这句话会报错
                image_tensor = image_tensor.to(device)
                out = model(image_tensor)
                # 得到预测结果，并且从大到小排序
                _, indices = torch.max(out, 1)

                filenode = {"filename" : 'test/'+realName, "label" : classes[indices]}
                filenodelist.append(filenode)

result='result.json'
with open(result,'w') as file_obj:
    json.dump(filenodelist,file_obj, sort_keys=True, indent=4, separators=(',', ': '))

