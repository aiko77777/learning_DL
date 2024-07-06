from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import os
img_path=(r"hymenoptera_data\train\ants\5650366_e22b7e1065.jpg")
image=Image.open(img_path)

writer=SummaryWriter("logs")    #tensorboard name
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(image)  #transform image to Tensor format 

writer.add_image("first_img",tensor_img)    #write in tensorboard
#writer.close()

print(tensor_img[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])    #normallize 
img_norm=trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image("normed_img",img_norm)
writer.close()