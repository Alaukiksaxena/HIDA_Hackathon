# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:22:16 2024

@author: a3536
"""

def generate_RGNIR_image(RGB_src_img_path,NIR_src_img_path):
    
    import cv2
    
    # assumes images are exactly the same size
    # safe in this case, but could include a test and/or a resize
    RGB_image_bgr = cv2.imread(RGB_src_img_path)
    print('RGB src image shape: ', RGB_image_bgr.shape)
    
    NIR_image_bgr = cv2.imread(NIR_src_img_path)
    print('NIR src image shape: ', NIR_image_bgr.shape)
    
    # Get the individual colour components of the images
    # NIR camera only uses 'Red' channel
    # break out the colour channels
    (B, G, R) = cv2.split(RGB_image_bgr)
    (BNIR, GNIR, RNIR) = cv2.split(NIR_image_bgr)
    
    # merge to form RGNIR image
    RGNIR_src_img_bgr = cv2.merge((RNIR,G,R))
    print('RGNIR src image shape: ', RGNIR_src_img_bgr.shape)

    return RGNIR_src_img_bgr