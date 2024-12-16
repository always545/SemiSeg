import torch.utils
import torch.utils.data
import torchvision.transforms.v2
from data import load_data
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torchvision
import PIL
import torch
from matplotlib import pyplot as plt
def rand_crop(img,lbl,height,width):
    i,j,h,w = RandomCrop.get_params(img,(height,width))
    return F.crop(img,i,j,h,w),F.crop(lbl,i,j,h,w)

def get_data_tensor(istrain = True):
    imgp,lblp = load_data('/root/semiseg/Dataset/VOCdevkit/VOC2012',istrain)
    img = [torchvision.io.read_image(imgname) for imgname in imgp]
    lbl = [torchvision.io.read_image(imgname) for imgname in lblp]
    return img,lbl


class Pascal_voc(torch.utils.data.Dataset):
    def __init__(self,img,lbl,size,supervised=True,is_train = True):
        super().__init__()
        self.is_train = is_train
        self.supervised = supervised
        self.img = img
        self.lbl = lbl
        self.size = size
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.crop = lambda img,lbl,size: rand_crop(img,lbl,*size)
        self.ctcrop = transforms.CenterCrop(size)
        self.strong_trans1 = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5,0.25),
            transforms.RandomGrayscale(0.2),
            transforms.GaussianBlur(3),
            transforms.RandomRotation(15)]
        )
        self.resize = transforms.Resize((300,300))
        self.strong_trans2 = transforms.Compose([  
            transforms.ColorJitter(0.5,0.5,0.5,0.25)
        ])
        self.weak_trans = transforms.Compose([
            transforms.RandomHorizontalFlip()]
        )
    def __len__(self):
        return len(self.img)
    
    def speturbtion(self,img):
        return self.strong_trans1(img),self.strong_trans2(img)

    def wpeturbtion(self,img):
        return self.weak_trans(img)
    
    def __getitem__(self, index):
        ignore_mask = torch.ones(self.size)
        if  self.is_train == False:
            return self.norm(self.img[index].to(torch.float)),self.lbl[index].to(torch.long).squeeze(0),index
        if not self.supervised:
            weak_img = self.norm(self.img[index].to(torch.float))
            weak_img = self.wpeturbtion(weak_img)
            strong_img1,strong_img2 = self.speturbtion(self.img[index])
            weak_img = self.ctcrop(weak_img)
            strong_img1 = self.ctcrop(strong_img1)
            strong_img2 = self.ctcrop(strong_img2)
            lbl = self.ctcrop(self.lbl[index]).squeeze(0).to(torch.long)
            ignore_mask[lbl==255] = 1
            lbl[lbl==255] = 0
            return weak_img,strong_img1,strong_img2,lbl,ignore_mask
        else :
            img = self.ctcrop(self.img[index]).to(torch.float)
            img = self.norm(img)
            lbl = self.ctcrop(self.lbl[index].squeeze(0).to(torch.long))
            ignore_mask[lbl==255] = 0
            lbl[lbl==255] = 0
            return img,lbl,ignore_mask
    


