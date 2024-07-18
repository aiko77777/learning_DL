import torchvision
from torch.utils.data import DataLoader


test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())               
img, target = test_data[0]
print(img.shape)
print(img)
#load data
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)      

for data in test_loader:
    imgs, targets = data   #output [4,3,32,32]  
    print(imgs.shape)
    print(targets)
