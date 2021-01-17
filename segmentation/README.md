# Semantic Segmentation
Semantic segmentation using RGB camera and Lidar scans
Dataset: [https://github.com/unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)

Before using:
* download the trained model and place in models folder: https://drive.google.com/file/d/1Vzrl9oMSX9Adv3sdVpGIDjAIEynNJ6Gn/view?usp=sharing
* replace the transform and intrinsic camera info files in config folder
* replace IMG_HEIGHT and IMG_WIDTH in transform_utils.py to match dimensions of camera input on robot
* replace c_crop in infer.py with desired dimensions of output

## Usage
Call infer(image, scan) from infer.py
* image is a HxWx3 rgb image as a numpy array
* scan is a lidar scan as a numpy array. can also be generated using load_from_bin in transform_utils.py

## Example
There are example files in the example folder. You can see how it works by running infer.py on it's own.
