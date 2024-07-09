import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(data,batch_size=64)

class Network(nn.Module):
    def __init__(self):
        super().__init__()  #調用父類的constructor
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        
    def forward(self,input):          # override forward()
        output = self.conv1(input)
        return output
    
network = Network()
tensorboard_writer=SummaryWriter("./log")
step=1
for datum in dataloader:  #dataloader from DataLoader is iterable
    img,label=datum
    output=network(img)
    output=torch.reshape(output,(-1,3,30,30))  #因為輸出是[64,6,30,30]==[batch_size,channel,height,width]，但tensorboard 只接受channel==3
    tensorboard_writer.add_images("input",img,step)
    tensorboard_writer.add_images("output",output,step)
    step+=1