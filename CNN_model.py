import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import imageio
import cv2
import os
import shutil
import glob
from tensorflow import keras
from imutils import paths
from PIL import Image


train_path_1 = "./UCSDped1/Train/Train*"
train_path_2 = "./UCSDped2/Train/Train*"

def copy_content(src_dir, dest_dir):
    files = os.listdir(src_dir)
    return shutil.copytree(src_dir, dest_dir)

a = 0
train_imgs = glob.glob(train_path_1 + "/*.tif", recursive = True)
#for i in train_imgs:
    #a+=1
#print(a)

for i in range(1):
	# plot raw pixel data
    im = cv2.imread(train_imgs[i])
    print(train_imgs[i])
    im_resized = cv2.resize(im, (238, 158), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    # define subplot
    plt.subplot(330 + 1 + i)
plt.show()
