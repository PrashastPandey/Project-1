import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2 as cv
import open3d as o3d
import matplotlib.pyplot as plt

import os
import random 
from pathlib import  Path

data_root=Path(".")
data_folder_path=data_root/"IndoorOutdoorimageclassification"
indoor_folder_path=data_folder_path/"indoor"
outdoor_folder_path=data_folder_path/"outdoor"

num_samples=3

selected_indoor_imgs=random.sample(os.listdir(indoor_folder_path),num_samples)
selected_outdoor_imgs=random.sample(os.listdir(outdoor_folder_path),num_samples)

indoor_imgs=[]
outdoor_imgs=[]

for i in range(num_samples):
    indoor_img=cv.imread(str(indoor_folder_path/selected_indoor_imgs[i]))
    indoor_img=cv.cvtColor(indoor_img,cv.COLOR_BGR2RGB)
    outdoor_img=cv.imread(str(outdoor_folder_path/selected_outdoor_imgs[i]))
    outdoor_img=cv.cvtColor(outdoor_img,cv.COLOR_BGR2RGB)

    indoor_imgs.append(indoor_img)
    outdoor_imgs.append(outdoor_img)

    print(indoor_img.shape,outdoor_img.shape)


processor=AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model=AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to("cuda")

print(model.config)

indoor_samples=[]
outdoor_samples=[]

for i in range(num_samples):
    indoor_inp=processor(images=indoor_imgs[i],return_tensors="pt").to("cuda")
    outdoor_inp=processor(images=outdoor_imgs[i],return_tensors="pt").to("cuda")

    with torch.no_grad():
        indoor_out=model(**indoor_inp)
        outdoor_out=model(**outdoor_inp)
        indoor_depth=indoor_out.predicted_depth
        outdoor_depth=outdoor_out.predicted_depth

    indoor_depth=indoor_depth.squeeze().cpu().numpy()
    outdoor_depth=outdoor_depth.squeeze().cpu().numpy()

    indoor_samples.append([indoor_imgs[i],indoor_depth])
    outdoor_samples.append([outdoor_imgs[i],outdoor_depth])


for i in range(num_samples):
    fig, axs=plt.subplots(1,2)

    axs[0].imshow(indoor_samples[i][0])
    axs[0].set_title("Indoor Img Orig")
    axs[1].imshow(indoor_samples[i][1])
    axs[1].set_title("Depth Img")
    plt.show()

    fig, axs=plt.subplots(1,2)
    
    axs[0].imshow(outdoor_samples[i][0])
    axs[0].set_title("Outdoor Img Orig")
    axs[1].imshow(outdoor_samples[i][1])
    axs[1].set_title("Depth Img")
    plt.show()

def get_intrinsics(H, W, fov=55.0):
    
    f=0.5*W/np.tan(0.5*fov*np.pi/180.0)
    cx=0.5*W
    cy=0.5*H
    return np.array([[f,0,cx],
                     [0,f,cy],
                     [0,0,1]])

def pixel2point(depth_img,camera_intrinsics=None):
    height, width=depth_img.shape
    if camera_intrinsics is None:
        camera_intrinsics=get_intrinsics(height,width,fov=55.0)

    fx, fy=camera_intrinsics[0,0], camera_intrinsics[1,1]
    cx, cy=camera_intrinsics[0,2], camera_intrinsics[1,2]

    x=np.linspace(0, width-1,width)
    y=np.linspace(0,height-1,height)

    u,v =np.meshgrid(x,y)

    x_over_z=(u-cx)/(fx)
    y_over_z=(v-cy)/(fy)

    z=depth_img/np.sqrt(1.+x_over_z**2+y_over_z**2)
    x=x_over_z*z
    y=y_over_z*z

    return x, y, z

def create_pointcloud(depth_img,color_img,camera_intrinsics=None, scale_ratio=100.0):
    height, width=depth_img.shape
    if camera_intrinsics is None:
        camera_intrinsics=get_intrinsics(height, width, fov=55.0)

    color_img=cv.resize(color_img, (width, height))

    depth_img=np.maximum(depth_img, 1e-5)
    depth_img=scale_ratio/depth_img

    x,y,z= pixel2point(depth_img, camera_intrinsics)
    point_img=np.stack((x,y,z), axis=-1)

    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(point_img.reshape(-1,3))
    cloud.colors=o3d.utility.Vector3dVector(color_img.reshape(-1,3)/255.0)

    return cloud

output_path=data_root/"point_clouds"
os.makedirs(output_path,exist_ok=True)

for i in range(num_samples):
    cloud =create_pointcloud(indoor_samples[i][1], indoor_samples[i][0])
    print(type(cloud))
    o3d.io.write_point_cloud(f"indoor_point_cloud_{i}.ply", cloud)

    cloud =create_pointcloud(outdoor_samples[i][1], outdoor_samples[i][0])
    o3d.io.write_point_cloud(f"outdoor_point_cloud_{i}.ply", cloud)
