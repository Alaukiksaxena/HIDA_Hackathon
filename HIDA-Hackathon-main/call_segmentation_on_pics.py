# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:13:20 2024

@author: a3536
"""

# https://github.com/facebookresearch/segment-anything visit site
# download model checkpoint: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# pip install git+https://github.com/facebookresearch/segment-anything.git, get segment-anything

# set path in segmentation_on_pics:
# checkpoint=r"d:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\segment-anything\sam_vit_h_4b8939.pth"

import numpy as np
from matplotlib import pyplot as plt
import os
from generate_RGNIR_image import generate_RGNIR_image
from PIL import Image
import numpy as np
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from segmentation_on_pics_light import segmentation_on_pics_light

data_path = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data'
path_data_train = data_path + r'\data_train' # change to your data folder with .npy files
# dirs created
path_data_train_masked = data_path + r'\data_train_masked' # files for further processes will be here!
path_pics_masked = data_path + r'\pics_masked' # leave bc it gets used later again
path_pics_rgb = data_path + r'\pics_rgb' # leave bc it gets used later again
path_pics_nir = data_path + r'\pics_nir' # leave bc it gets used later again
path_pics_rgnir = data_path + r'\pics_rgnir' # leave bc it gets used later again

if not os.path.isdir(path_data_train_masked):
    os.mkdir(path_data_train_masked)
if not os.path.isdir(path_pics_masked):
    os.mkdir(path_pics_masked)
if not os.path.isdir(path_pics_rgb):
    os.mkdir(path_pics_rgb)
if not os.path.isdir(path_pics_nir):
    os.mkdir(path_pics_nir)
if not os.path.isdir(path_pics_rgnir):
    os.mkdir(path_pics_rgnir)

file_extension = '.npy'
all_files = [f for f in os.listdir(path_data_train) if os.path.isfile(os.path.join(path_data_train, f)) and f.endswith(file_extension)]

for file_name in all_files:
    path_npy = path_data_train + '\\' + file_name
    print(file_name)
    data = np.load(path_npy)
    
    path_pics_rgb_file = path_pics_rgb + '\\' + file_name[:-4] + '.jpg'
    if not os.path.isfile(path_pics_rgb_file):
        image_RGB = data[:,:,:3]
        image = Image.fromarray(image_RGB.astype('uint8')).convert('RGB')
        image.save(path_pics_rgb_file)
    
    path_pics_nir_file = path_pics_nir + '\\' + file_name[:-4] + '.jpg'
    if not os.path.isfile(path_pics_nir_file):
        image_NIR = data[:,:,3]
        image = Image.fromarray(image_NIR.astype('uint8')).convert('RGB')
        image.save(path_pics_nir_file)
    
    RGNIR_src_img_bgr = generate_RGNIR_image(path_pics_rgb_file,path_pics_nir_file)
    path_pics_rgnir_file = path_pics_rgnir + '\\' + file_name[:-4] + '.jpg'
    if not os.path.isfile(path_pics_rgnir_file):
        image_RGB = RGNIR_src_img_bgr
        image = Image.fromarray(image_RGB.astype('uint8')).convert('RGB')
        image.save(path_pics_rgnir_file)
    print('done creating .jpg files from .npv for RGB and NIR and RGNIR')
    
    anns = []
    path_pics_masked_file = path_pics_masked + '\\' + file_name[:-4] + '.jpg'
    path_data_train_masked_file = path_data_train_masked + '\\' + file_name[:-4] + '.npy'
    if not os.path.isfile(path_pics_masked_file) or not os.path.isfile(path_data_train_masked_file):
        anns = segmentation_on_pics_light(RGNIR_src_img_bgr,path_pics_masked_file)
        # if len(anns) == 0:
        #     return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            # color_mask = np.concatenate([np.random.random(3), [0.35]])
            color_mask = np.random.random(1)
            img[m] = color_mask
        newdata = np.concatenate([data,img[:,:,0:1]],axis=2)
        path_data_train_masked_file = path_data_train_masked + '\\' + file_name[:-4] + '.npy'
        with open(path_data_train_masked_file, 'wb') as f:
            np.save(f, newdata)
    else:
        pass
    # asx

#%% segmentation
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from segmentation_on_pics import segmentation_on_pics

# path_to_pics_folder = path_pics_train
# file_extension = '.jpg'

# masks = segmentation_on_pics(data_path,path_to_pics_folder,file_extension,plots=True)
