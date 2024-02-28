# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:37:45 2024

@author: a3536
"""

import pandas as pd
import numpy as np
import seaborn as sns
import sys
from matplotlib import pyplot as plt
import os
sys.path.append('d:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\HIDA-Hackathon-main')

#%% paths

path_train = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\data_train'

path = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\data_train\DJI_0003_R.npy'

# data = np.load(path)

# pic0 = pd.DataFrame(data[:,:,0]) #R
# pic1 = pd.DataFrame(data[:,:,1]) #G
# pic2 = pd.DataFrame(data[:,:,2]) #B
# pic3 = pd.DataFrame(data[:,:,3]) #Heat
# pic4 = pd.DataFrame(data[:,:,4]) #depth

# # sns.heatmap(pic0)
# # sns.heatmap(pic1)
# # sns.heatmap(pic2)
# # sns.heatmap(pic3)
# # sns.heatmap(pic4)
# plt.imshow(data[:,:,0:3])

#%% show images
folder_path = path_train
file_extension = '.npy'
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)]

for file_name in all_files:
    path = folder_path + '\\' + file_name
    print(file_name)
    data = np.load(path)
    # plt.imshow(data[:,:,0:3])
    # temp = data[:,:,0]
    # data[:,:,0] = data[:,:,1]
    # data[:,:,1] = temp
    plt.imshow(data[:,:,:3])
    plt.axis('off')
    # plt.show()
    path_figs = rf'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\pics_train\{file_name[:-4]}.jpg'
    plt.savefig(path_figs,format='jpg', bbox_inches='tight')
    plt.close()


#%%
# from dataset import DroneImages
# x,y = DroneImages(path_train)

#%%
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torchvision
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#%%
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

path_fig = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\pics_train\DJI_0003_R.jpg'

image = cv2.imread(path_fig)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

# sam_checkpoint = "sam_vit_h_4b8939.pth"
checkpoint=r"d:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\segment-anything\sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

#%%
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks2 = mask_generator_2.generate(image)
len(masks2)
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show()

#%% function for segmentation

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torchvision
os.environ['KMP_DUPLICATE_LIB_OK']='True'
path_fig = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\pics_train\DJI_0003_R.jpg'

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def call_segmentation_on_pic(path_to_pics_folder,file_extension,plots=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    all_files = [f for f in os.listdir(path_to_pics_folder) if os.path.isfile(os.path.join(path_to_pics_folder, f)) and f.endswith(file_extension)]
    
    for file in all_files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if plots:
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    # sam checkpoint file must be available!
    checkpoint=r"d:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\segment-anything\sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    masks = mask_generator_2.generate(image)

    if plots:
        import matplotlib.pyplot as plt
        print(len(masks))
        print(masks[0].keys())
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show()
        
#%% test
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from segmentation_on_pics import segmentation_on_pics

path_to_pics_folder = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\pics_train'
file_extension = '.jpg'

masks = segmentation_on_pics(path_to_pics_folder,file_extension,plots=True)