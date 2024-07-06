from torch.utils.data import Dataset
from PIL import Image
import os
"""
load data from Dataset
"""

class MyData(Dataset):     
    def __init__(self,root_dir,label_dir):    #constructor
        self.root_dir = root_dir 
        self.label_dir = label_dir     
        self.path = os.path.join(self.root_dir,self.label_dir) #join 串接
        self.img_path = os.listdir(self.path) #把dir裡內容轉成list
        
    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)            
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
print(len(ants_dataset))    #length of ants dataset 
print(len(bees_dataset))    #length of bees dataset
train_dataset = ants_dataset + bees_dataset  #union  
print(len(train_dataset))   #length of union set

img,label = train_dataset[200]
print("label：",label)
img.show()