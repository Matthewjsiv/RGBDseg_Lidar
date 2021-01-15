from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import time
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm


with open('Rellis_3D_ontology/ontology.yaml') as f:
    DATA = yaml.load(f)

class SegDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, imageFolder, maskFolder):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
        """
        self.root_dir = root_dir
        self.image_names = sorted(
            glob.glob(os.path.join(self.root_dir, '*', 'in.jpg')))
        self.mask_names = sorted(
            glob.glob(os.path.join(self.root_dir, '*', 'target.png')))
        self.scan_names = sorted(
            glob.glob(os.path.join(self.root_dir, '*', 'depth.npy')))
        assert len(self.image_names) == len(self.scan_names) and len(self.image_names) == len(self.mask_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        scan_name = self.scan_names[idx]
        c_crop = (1024,1024)

        input_image = Image.open(img_name)
        preprocess = transforms.Compose([
            transforms.CenterCrop(c_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        image = input_tensor

        scan = np.load(scan_name)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(c_crop)

        ])
        input_scan = preprocess(scan)
        image = torch.cat([image,input_scan])

        #we also want the normal image for visualization purposes
        preprocess = transforms.Compose([
            transforms.CenterCrop(c_crop),
            transforms.ToTensor()])
        nimage = preprocess(input_image)


        msk_name = self.mask_names[idx]
        mask = Image.open(msk_name)
        preprocess = transforms.Compose([
            transforms.CenterCrop(c_crop),
            transforms.ToTensor()
        ])
        mask = preprocess(mask) * 255
        mask = mask.to(torch.uint8)
        mask = mask.squeeze()

        #convert keys 0:34 to 0:19
        convert = DATA[1]
        for elem in convert:
            mask[mask==elem] = convert[elem]


        sample = {'image': image, 'mask': mask, 'nimage': nimage, 'scan':input_scan}
        return sample

def DeepLabModel(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet50(
        pretrained=True, progress=True)

    model.classifier = DeepLabHead(2048, outputchannels)
    return model

def DeepLabModel_rgbd(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet50(
        pretrained=True, progress=True)
    #model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model

#you can vizualize different directories by changing the directory names here
dataset = SegDataset('ndataset/train','input','targets')
dataloader = DataLoader(dataset, batch_size=2)
num_items = len(dataset)

model = DeepLabModel_rgbd(20)
def convert_batchnorm_to_instancenorm(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, torch.nn.Identity())
            #print(child.num_features)
        else:
            convert_batchnorm_to_instancenorm(child)

convert_batchnorm_to_instancenorm(model)
model.load_state_dict(torch.load("checkpoints/identity4/checkpoint-1-9"))
model.eval()
model.cuda()

for child in model.children():
    print(child)
    break



writer = SummaryWriter('vizboard')

with torch.no_grad():
    idx = 0
    for sample in tqdm(dataset,total=num_items):
        criterion = torch.nn.CrossEntropyLoss()
        fig=plt.figure(dpi=400)
        nimage = sample['nimage'].transpose(0,2).transpose(0,1)
        #writer.add_image('sample'+ str(idx) + '/image',nimage,idx,dataformats='CHW')

        fig.add_subplot(1, 4, 1).title.set_text('Depth')
        scan = np.squeeze(sample['scan'])
        plt.imshow(scan)
        plt.axis('off')

        fig.add_subplot(1, 4, 2).title.set_text('Image')
        plt.imshow(nimage)
        plt.axis('off')

        mask = sample['mask']
        nmask = np.zeros((1024,1024,3),dtype=np.int32)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                key = mask[i,j].byte().cpu().item()
                pallette = DATA[2][key]

                nmask[i,j,0] = pallette[0]
                nmask[i,j,1] = pallette[1]
                nmask[i,j,2] = pallette[2]
        #writer.add_image('sample'+ str(idx) + '/GT',nmask,idx,dataformats='HWC')
        fig.add_subplot(1, 4, 3).title.set_text('GT')
        plt.imshow(nmask)
        plt.axis('off')

        input = sample['image'].unsqueeze(0).cuda()
        #print(input.shape)
        output = model(input)
        # toutput = output.copy()
        # toutput = toutput['out']
        # toutput = torch.flatten(toutput,start_dim=2,end_dim=3)
        # masks = torch.flatten(sample['mask'].unsqueeze(0).cuda(),start_dim=1)
        # masks = masks.to(torch.int64)
        # loss = criterion(toutput,masks)
        # print(loss)
        output = output['out'][0]

        output = output.argmax(0)
        # print(output)
        # plot the semantic segmentation predictions of 21 classes in each color
        output = output.byte().cpu()
        output.unsqueeze_(-1)
        output = output.expand(1024,1024,3)
        output = output.numpy()
        output = output.astype(np.int32)
        convert = DATA[1]
        data = DATA[2]
        #print(mask)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                key = output[i,j,0]
                pallette = data[key]

                output[i,j,0] = pallette[0]
                output[i,j,1] = pallette[1]
                output[i,j,2] = pallette[2]
        #writer.add_image('sample'+ str(idx) + '/Prediction',output,idx,dataformats='HWC')
        fig.add_subplot(1, 4, 4).title.set_text('Prediction')
        #plt.imshow(output)
        plt.axis('off')
        #print(output)
        idx +=1
        plt.imshow(output)
        plt.savefig('rgbdviz/train/' + str(idx) + '.png')
        plt.close(fig)
writer.close()
