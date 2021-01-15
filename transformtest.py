from IPython.display import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import matplotlib.image as mpimg
import yaml
#%matplotlib inline



def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:3]



def print_projection_plt_depth(points, color, image_width, image_height):
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grey_image = 0 * np.ones((image_height,image_width), dtype = np.float32)
    for i in range(points.shape[1]):
        cv2.circle(grey_image, (np.float32(points[0][i]),np.float32(points[1][i])),3, (int(color[i])),-1)

    #grey_image = cv2.GaussianBlur(grey_image,(5,5),0)

    return grey_image

def print_projection_plt(points, color, image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def depth_color(val, min_d=0, max_d=120):
    np.clip(val, 0, max_d, out=val)
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)
def points_filter(points,img_width,img_height,P,RT):
    ctl = RT
    ctl = np.array(ctl)
    fov_x = 2*np.arctan2(img_width, 2*P[0,0])*180/3.1415926+10
    fov_y = 2*np.arctan2(img_height, 2*P[1,1])*180/3.1415926+10
    R= np.eye(4)
    p_l = np.ones((points.shape[0],points.shape[1]+1))
    p_l[:,:3] = points
    p_c = np.matmul(ctl,p_l.T)
    p_c = p_c.T
    x = p_c[:,0]
    y = p_c[:,1]
    z = p_c[:,2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    xangle = np.arctan2(x, z)*180/np.pi;
    yangle = np.arctan2(y, z)*180/np.pi;
    flag2 = (xangle > -fov_x/2) & (xangle < fov_x/2)
    flag3 = (yangle > -fov_y/2) & (yangle < fov_y/2)
    res = p_l[flag2&flag3,:3]
    res = np.array(res)
    x = res[:, 0]
    y = res[:, 1]
    z = res[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    color = depth_color(dist, 0, 70)
    return res,color



def get_cam_mtx(filepath):
    data = np.loadtxt(filepath)
    P = np.zeros((3,3))
    P[0,0] = data[0]
    P[1,1] = data[1]
    P[2,2] = 1
    P[0,2] = data[2]
    P[1,2] = data[3]
    return P

def get_lidar2cam_mtx(filepath):
    with open(filepath,'r') as f:
        data = yaml.load(f,Loader= yaml.Loader)
    q = data['os1_cloud_node-pylon_camera_node']['q']
    q = np.array([q['w'],q['x'],q['y'],q['z']])
    t = data['os1_cloud_node-pylon_camera_node']['t']
    t = np.array([t['x'],t['y'],t['z']])
    R_vc = Rotation(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4,4)
    RT[:3,:3] = R_vc
    RT[:3,-1] = t
    RT = np.linalg.inv(RT)
    R= np.eye(3)
    R[0,0]=-1
    R[1,1]=-1
    RT[:-1,:-1]=np.matmul(R,RT[:-1,:-1])
    return RT

def get_im(im_name,lidarname,series):
    #print(im_name)
    image = cv2.imread(im_name)
    points = load_from_bin(lidarname)
    img_height, img_width, channels = image.shape
    distCoeff = np.array([-0.134313,-0.025905,0.002181,0.00084,0])
    distCoeff = distCoeff.reshape((5,1))
    P = get_cam_mtx('config/Rellis_3D_cam_intrinsic/Rellis-3D/' + str(series) + '/camera_info.txt')
    #print(P)



    RT= get_lidar2cam_mtx('config/Rellis_3D_cam2lidar/Rellis-3D/' + str(series) + '/transforms.yaml')
    R2= np.eye(3)
    theta = -np.pi/400
    R2[0,0]= np.cos(theta)
    R2[1,1]=R2[0,0]
    R2[0,1]= np.sin(theta)
    R2[1,0]=-R2[0,1]
    RT[:-1,:-1]=np.matmul(RT[:-1,:-1],R2)
    R_vc = RT[:3,:3]
    T_vc = RT[:3,3]
    T_vc = T_vc.reshape(3, 1)
    rvec,_ = cv2.Rodrigues(R_vc)
    tvec = T_vc
    xyz_v, c_ = points_filter(points,img_width,img_height,P,RT)

    imgpoints, _ = cv2.projectPoints(xyz_v[:,:],rvec, tvec, P, distCoeff)
    imgpoints = np.squeeze(imgpoints,1)
    imgpoints = imgpoints.T
    #print(imgpoints.shape)
    #print(image.shape)
    res = print_projection_plt(points=imgpoints, color=c_, image=image)

    # display result image
    plt.subplots(1,1, figsize = (20,20) )
    plt.title("Velodyne points to camera image Result")
    plt.imshow(res)
    plt.show()
def get_depth(im_name,lidarname,series):
    #image = cv2.imread(im_name)
    points = load_from_bin(lidarname)
    #img_height, img_width, channels = image.shape
    img_height = 1200
    img_width = 1920
    distCoeff = np.array([-0.134313,-0.025905,0.002181,0.00084,0])
    distCoeff = distCoeff.reshape((5,1))
    P = get_cam_mtx('config/Rellis_3D_cam_intrinsic/Rellis-3D/' + str(series) + '/camera_info.txt')
    #print(P)



    RT= get_lidar2cam_mtx('config/Rellis_3D_cam2lidar/Rellis-3D/' + str(series) + '/transforms.yaml')
    R2= np.eye(3)
    theta = -np.pi/400
    R2[0,0]= np.cos(theta)
    R2[1,1]=R2[0,0]
    R2[0,1]= np.sin(theta)
    R2[1,0]=-R2[0,1]
    RT[:-1,:-1]=np.matmul(RT[:-1,:-1],R2)
    R_vc = RT[:3,:3]
    T_vc = RT[:3,3]
    T_vc = T_vc.reshape(3, 1)
    rvec,_ = cv2.Rodrigues(R_vc)
    tvec = T_vc
    xyz_v, c_ = points_filter(points,img_width,img_height,P,RT)

    imgpoints, _ = cv2.projectPoints(xyz_v[:,:],rvec, tvec, P, distCoeff)
    imgpoints = np.squeeze(imgpoints,1)
    imgpoints = imgpoints.T
    #print(imgpoints.shape)
    #print(image.shape)
    res = print_projection_plt_depth(points=imgpoints, color=c_, image_width = img_width, image_height=img_height)

    # display result image
    plt.subplots(1,1, figsize = (20,20) )
    plt.title("Velodyne points to camera image Result")
    plt.imshow(res)
    plt.show()
    return res
