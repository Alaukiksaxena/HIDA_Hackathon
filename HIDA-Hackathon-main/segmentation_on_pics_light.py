# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:44:23 2024

@author: a3536
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torchvision
import os

# path_fig = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\pics_train\DJI_0003_R.jpg'

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

def segmentation_on_pics_light(image,path_pics_masked_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
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

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    # plt.show()
    plt.savefig(path_pics_masked_file,format='jpg', bbox_inches='tight')
    plt.close()
    
    # sizes = np.shape(image)     
    # fig = plt.figure()
    # fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(image)
    # show_anns(masks)
    # # plt.savefig(path_pics_masked_file,format='jpg', bbox_inches='tight')
    # plt.savefig(path_pics_masked_file,format='jpg', dpi = sizes[0]) 
    # plt.close()
    return masks