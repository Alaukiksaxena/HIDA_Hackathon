# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:13:20 2024

@author: a3536
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from segmentation_on_pics import segmentation_on_pics

path_to_pics_folder = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\pics_train'
file_extension = '.jpg'

masks = segmentation_on_pics(path_to_pics_folder,file_extension,plots=True)