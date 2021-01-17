from transform_utils import load_from_bin, get_depth
import torch
from torchvision import transforms
import numpy as np
from model import DeepLabModel_rgbd, convert_batchnorm_to_instancenorm

#model init
model = DeepLabModel_rgbd(20)
convert_batchnorm_to_instancenorm(model)
model.load_state_dict(torch.load("models/checkpoint-1-9"))
model.eval()
model.cuda()

def infer(image,scan):
    """Image is a numpy array - HxWx3 (H- height, W- width, 3- RGB)
    Scan is a lidar scan in numpy format Nx3 (N- number of points)
    Scan can be created using load_from_bin in transform_utils
    Output: 1024x1024 mask"""
    #size of image going into model
    c_crop = (1024,1024)

    input_image = image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(c_crop),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    image = input_tensor

    points = get_depth(scan)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(c_crop)

    ])
    input_scan = preprocess(points)
    image = torch.cat([image,input_scan])

    input = image.unsqueeze(0).cuda()
    output = model(input)
    output = output['out'][0]
    output = output.argmax(0)
    output = output.byte().cpu()
    # output.unsqueeze_(-1)
    # output = output.expand(c_crop[0],c_crop[1],3).numpy().astype(np.int32)
    output = output.numpy().astype(np.int32)

    return output

def test():
    from PIL import Image
    import matplotlib.pyplot as plt

    scan = load_from_bin('example/000104.bin')
    # points = get_depth(scan)
    # plt.imshow(points)
    # plt.show()
    # image = cv2.imread('example/frame000104-1581624663_149.jpg')
    # image = get_im('example/frame000104-1581624663_149.jpg',scan)
    # plt.imshow(image)
    # plt.show()
    image = Image.open('example/frame000104-1581624663_149.jpg')
    image = np.array(image)
    print(image.shape)
    print(scan.shape)
    output = infer(image,scan)
    print(output.shape)
    plt.imshow(output*10)
    plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    test()
