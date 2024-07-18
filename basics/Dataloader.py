import torchvision
from torch.utils.data import DataLoader


test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())               
img, target = test_data[0]
print(img.shape)
print(img)

# batch_size=4 使得 img0, target0 = dataset[0]、img1, target1 = dataset[1]、img2, target2 = dataset[2]、img3, target3 = dataset[3]，然后这四个数据作为Dataloader的一个返回      
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)      

for data in test_loader:
    imgs, targets = data            
    print(imgs.shape)
    print(targets)
