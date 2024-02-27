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
sys.path.append('d:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\HIDA-Hackathon-main')

path = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\data_train\DJI_0003_R.npy'
path_train = r'd:\Profile\a3536\Eigene Dateien\GitHub\HIDA_Hackathon\data\data_train'

data = np.load(path)

pic0 = pd.DataFrame(data[:,:,0]) #R
pic1 = pd.DataFrame(data[:,:,1]) #G
pic2 = pd.DataFrame(data[:,:,2]) #B
pic3 = pd.DataFrame(data[:,:,3]) #Heat
pic4 = pd.DataFrame(data[:,:,4]) #depth

# sns.heatmap(pic0)
# sns.heatmap(pic1)
# sns.heatmap(pic2)
# sns.heatmap(pic3)
# sns.heatmap(pic4)
plt.imshow(data[:,:,0:3])


#%%
from dataset import DroneImages
x,y = DroneImages(path_train)

#%%
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(pic3)