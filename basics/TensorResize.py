from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

img_path = r"basics\hymenoptera_data\val\bees\6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"
img = Image.open(img_path)
print(img)  

writer = SummaryWriter("logs") 

trans_totensor = transforms.ToTensor() 
img_tensor = trans_totensor(img)  

trans_resize = transforms.Resize((512,512))

img_resize = trans_resize(img)  #resize to 512x512
img_resize = trans_totensor(img_resize)

print(img_resize.size()) # output: 3×512×512，3通道

writer.add_image("img_tensor",img_tensor) 
writer.add_image("img_resize",img_resize) 
writer.close()
