import torch
import torch.nn
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(data,batch_size=64)
writer=SummaryWriter("maxpool_logs")
class Network(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_pool=MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self,input):
        output=self.max_pool(input)
        return output
    

network=Network()

step=1
for datum in dataloader:
    img,label=datum
    writer.add_images("unmaxpool",img,step)
    img=network(img)
    writer.add_images("maxpool",img,step)
    step+=1

writer.close()