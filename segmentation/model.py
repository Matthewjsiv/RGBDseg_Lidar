from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch

def DeepLabModel_rgbd(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet50(
        pretrained=True, progress=True)
    #model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    #model.load_state_dict(torch.load("checkpoints/freeze-8_oldish-5/checkpoint-20000"))
    model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model

def convert_batchnorm_to_instancenorm(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, torch.nn.Identity())
            #print(child.num_features)
        else:
            convert_batchnorm_to_instancenorm(child)
