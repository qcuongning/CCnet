import cv2
import os
import numpy as np
def Loader(img_path,mask_path,target_size):
    im_n = os.listdir(img_path)
    imgs= []
    masks = []
    for n in im_n[:100]:
        img =cv2.imread(os.path.join(img_path,n))
        mn = "mask_"+n
        mask = cv2.imread((os.path.join(mask_path,mn)),cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            img = cv2.resize(img,target_size)
            mask = cv2.resize(mask,target_size)
            img = img /255
           # mask[mask<1] = 0
           # mask[mask>=1] = 1
            mask = mask/255
            imgs.append(img)
            masks.append(mask)
    imgs = np.asarray(imgs, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    return (imgs,masks)