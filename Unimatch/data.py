import torchvision
import os
def download_data():
    data = torchvision.datasets.VOCDetection(
        root= os.path.join('Dataset/'),
        year='2012',
        image_set='train',
        download=True
    )

def load_data(root,is_train):
    if is_train:
        filepath = os.path.join(root,'ImageSets','Segmentation','train.txt')
    else:
        filepath = os.path.join(root,'ImageSets','Segmentation','val.txt')
    with open(filepath,'r') as f:
        train_data = f.read().split()
    images = [os.path.join(root,'JPEGImages',i+'.jpg') for i in train_data]
    labels = [os.path.join(root,'SegmentationClass',i+'.png') for i in train_data]
    return images,labels

if __name__ == '__main__':
    download_data()