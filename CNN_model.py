import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import cv2
import os
import shutil
import glob
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from imutils import paths
import matplotlib.image as mpimg
from tensorflow.python.client import device_lib 


#Path for training set of Ped1 and Ped2
train_path_1 = "./UCSDped1/Train/Train*"
train_path_2 = "./UCSDped2/Train/Train*"

#Path for test set of Ped1 and Ped2
test_path_1 = "./UCSDped1/Test/Test*"
test_path_2 = "./UCSDped2/Test/Test*"

#Image dimension of Ped1 and Ped2
img_width_ped1 = 238
img_height_ped1 = 158

img_width_ped2 = 360
img_height_ped2 = 240

#Print the total images inside the set
def count_images(path):
    a = 0
    train_imgs = glob.glob(path + "/*.tif", recursive = True)
    for i in train_imgs:
        a+=1
    return a

#Return the list of all images inside "Train" or "Test" files of 2 sets Ped1 and Ped2
def image_path_list(path):
    train_imgs = glob.glob(path + "/*.tif", recursive = True)
    return train_imgs

#Show 9 example images
def show_9_example_imgs(imgs, img_width, img_height):
    plt.figure(figsize=(20,20))
    nine_random_imgs = np.random.choice(imgs, 9, replace=False)
    for i in range(9):
        im = cv2.imread(nine_random_imgs[i])
        im_resized = cv2.resize(im, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
        # define subplot
        plt.subplot(330 + 1 + i)
    return plt.show()

#Create dataset and convert its images to value between 0 and 1
def create_dataset(img_folder, img_quantity, img_width, img_height):
    empty_array = np.empty((img_quantity, img_width, img_height))
    for dir1 in range(len(img_folder)):
        image = cv2.imread(img_folder[dir1], cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_height, img_width),interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float64')
        image /= 255 
        empty_array[dir1, :, :] = image
    return empty_array

#Create labels for training set
def training_labels(img_quantity):
    return np.zeros((img_quantity, 1))

#Autoencoder
def make_convolutional_autoencoder(shape):
    # encoding
    inputs = Input(shape)
    x = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D( 8, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D( 8, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling2D(padding='same')(x)    
    
    # decoding
    x = Conv2D( 8, 3, activation='relu', padding='same')(encoded)
    x = UpSampling2D()(x)
    x = Conv2D( 8, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(16, 3, activation='relu')(x) 
    x = UpSampling2D()(x)
    x = Conv2D(16, 3, activation='relu')(x) 
    x = UpSampling2D()(x)
    decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)    
    
    # autoencoder
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


#MAIN
def main():
    #create a list contain all path of images
    train_ped1 = image_path_list(train_path_1)
    train_ped2 = image_path_list(train_path_2)

    #print total images in "Train" of UCSDPed1
    #print(count_images(train_path_1))

    #show 9 example images in Ped1 and Ped 2
    #show_9_example_imgs(train_ped1, img_width_ped1, img_height_ped1)

    #Training label for Ped1 and Ped2
    train_ped1_label = training_labels(count_images(train_path_1))
    train_ped2_label = training_labels(count_images(train_path_2))

    #Create training dataset for Ped1 and Ped2
    train_ped1_dataset = create_dataset(train_ped1, count_images(train_path_1), img_width_ped1 ,img_height_ped1)
    train_ped1_dataset.reshape(-1, img_width_ped1 ,img_height_ped1, 1)
    #print(train_ped1_dataset.shape)
    train_ped2_dataset = create_dataset(train_ped2, count_images(train_path_2), img_width_ped2 ,img_height_ped2)
    train_ped2_dataset.reshape(-1, img_width_ped2 ,img_height_ped2, 1)
    #print(train_ped2_dataset.shape)

    #print(device_lib.list_local_devices())

    #Call the autoencoder
    autoencoder_ped1 = make_convolutional_autoencoder(shape=(img_width_ped1, img_height_ped1, 1))
    #autoencoder_ped1.summary()
    #Fitting
    autoencoder_ped1.fit(train_ped1_dataset, train_ped1_label, epochs=50, batch_size=128)



if __name__ == '__main__':
    main()
