import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import cv2
import glob
import matplotlib.image as mpimg
import PIL
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from imutils import paths
from PIL import Image
from tensorflow.python.client import device_lib 
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

#Declaration
#Path for training set of Ped1 and Ped2
train_path_1 = ".\\UCSDped1\\Train\\Train*"
train_path_2 = ".\\UCSDped2\\Train\\Train*"

#Path for test set of Ped1 and Ped2
test_path_1 = ".\\UCSDped1\\Test\\Test*"
root_path_1 = ".\\UCSDped1\\Test"
test_path_2 = ".\\UCSDped2\\Test\\Test*"
root_path_2 = ".\\UCSDped2\\Test"


#Files in test set which contains anomaly
anomaly_ped1 = ["\\Test003", "\\Test004", "\\Test014", "\\Test018", "\\Test019", "\\Test021", "\\Test022", "\\Test023", "\\Test024", "\\Test032"]
anomaly_ped2 = ["\\Test001", "\\Test002", "\\Test003", "\\Test004", "\\Test005", "\\Test006", "\\Test007", "\\Test008", "\\Test009", "\\Test010", "\\Test011", "\\Test012"]

#Image dimension of Ped1 and Ped2\adding-new-column-to-existing-dataframe-in-pandas\
img_width_ped1 = 238
img_height_ped1 = 158
#The desired image size of Ped1 for the ease of autoencoder
img_width_ped1_desire = 236
img_height_ped1_desire = 156

img_width_ped2 = 360
img_height_ped2 = 240
#The desired image size of Ped2 for the ease of autoencoder
img_width_ped2_desire = 364
img_height_ped2_desire = 236


#Functions
#Print the total images inside the set
def count_images(path):
    a = 0
    imgs = glob.glob(path + "\\*.tif", recursive = True)
    for i in imgs:
        a+=1
    return a
#Return the list of all images inside "Train" or "Test" files of 2 sets Ped1 and Ped2
def image_path_list(path):
    imgs = glob.glob(path + "\\*.tif", recursive = True)
    return imgs
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
#Create labels with all ones, meaning normal = 1
def training_labels(img_quantity):
    labels = np.ones((img_quantity, 1)) 
    return labels
#Creating labels for test set
def test_labels(img_quantity, root_path, anomaly_set, test_ped):
    labels = np.ones((img_quantity, 1))
    ped = [root_path + s for s in anomaly_set]
    list = []
    for i in ped:
        anomaly = glob.glob(i + "\\*.tif", recursive = True)
        for a in anomaly:
            list.append(a)
    for b in range(img_quantity): 
        if (test_ped[b] in list):
            labels[b] = 0
    return labels
#Show images before and after being coded
def show_images(before_images, after_images):
    plt.figure(figsize=(10, 2))
    for i in range(10):
        # before
        plt.subplot(2, 10, i+1)
        plt.imshow(before_images[i].reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        # after
        plt.subplot(2, 10, 10+i+1)
        plt.imshow(after_images[i].reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()
#Plotting loss
def loss_plot(ped):
    plt.plot(ped.history["loss"], label="Training Loss")
    plt.plot(ped.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    return plt.show()
#Autoencoder
def make_convolutional_autoencoder(shape):
    # encoding
    inputs = Input(shape)
    x = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
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
    x = Conv2D(16, 3, activation='relu', padding='same')(x) 
    x = UpSampling2D()(x)
    x = Conv2D(16, 3, activation='relu')(x) 
    x = UpSampling2D()(x)
    decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)    
    
    # autoencoder
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
#MSE
def mean_squared_error(reconstructed, original):
    tensor4d_mse = (reconstructed - original) ** 2 #tensor 4d
    empt_arr = np.zeros((tensor4d_mse.shape[0], 1))
    #Take the largest loss value in each image, return it into array
    for i in range(tensor4d_mse.shape[0]):
        empt_arr[i] = np.amax(tensor4d_mse[i, :, :])
    return empt_arr
#Predict
def predict(model, data, threshold):
  reconstructions = model.predict(data)
  loss = mean_squared_error(reconstructions, data)
  return tf.math.less(loss, threshold)
#Statics
def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))



#MAIN
def main():
    #create a list contain all path of images
    train_ped1 = image_path_list(train_path_1)
    test_ped1 = image_path_list(test_path_1)

    train_ped2 = image_path_list(train_path_2)
    test_ped2 = image_path_list(test_path_2)

    #print total images in "Train" of UCSDPed1
    #print(count_images(train_path_1))

    #show 9 example images in Ped1 and Ped 2
    #show_9_example_imgs(train_ped1, img_width_ped1, img_height_ped1)

    #Training label for Ped1 and Ped2
    train_ped1_label = training_labels(count_images(train_path_1))
    train_ped2_label = training_labels(count_images(train_path_2))
    #Test lebel for Ped1 and Ped2
    test_ped1_label = test_labels(count_images(test_path_1), root_path_1, anomaly_ped1, test_ped1)
    test_ped2_label = test_labels(count_images(test_path_2), root_path_2, anomaly_ped2, test_ped2)

    #Create training dataset for Ped1 and Ped2
    #The size of image now change to 236x156 in Ped1 and 364x236 in Ped2
    train_ped1_dataset = create_dataset(train_ped1, count_images(train_path_1), img_width_ped1_desire ,img_height_ped1_desire)
    train_ped1_dataset = train_ped1_dataset.reshape(count_images(train_path_1), img_width_ped1_desire ,img_height_ped1_desire, 1)
    #print(train_ped1_dataset.shape)
    train_ped2_dataset = create_dataset(train_ped2, count_images(train_path_2), img_width_ped2_desire ,img_height_ped2_desire)
    train_ped2_dataset = train_ped2_dataset.reshape(count_images(train_path_2), img_width_ped2_desire ,img_height_ped2_desire, 1)
    #print(train_ped2_dataset.shape)

    #Create test dataset for Ped1 and Ped2
    test_ped1_dataset = create_dataset(test_ped1, count_images(test_path_1), img_width_ped1_desire ,img_height_ped1_desire)
    test_ped1_dataset = test_ped1_dataset.reshape(count_images(test_path_1), img_width_ped1_desire ,img_height_ped1_desire, 1)
    #print(test_ped1_dataset.shape)
    test_ped2_dataset = create_dataset(test_ped2, count_images(test_path_2), img_width_ped2_desire ,img_height_ped2_desire)
    test_ped2_dataset = test_ped2_dataset.reshape(count_images(test_path_2), img_width_ped2_desire,img_height_ped2_desire, 1)
    #print(test_ped2_dataset.shape)

    #print(device_lib.list_local_devices())

    #Call the autoencoder
    #autoencoder_ped1 = make_convolutional_autoencoder(shape=(img_width_ped1, img_height_ped1, 1))
    autoencoder_ped1 = make_convolutional_autoencoder(shape=(img_width_ped1_desire, img_height_ped1_desire, 1))
    #autoencoder_ped1.summary()

    #Fitting
    #autoencoder_ped1.fit(train_ped1_dataset, train_ped1_dataset, epochs=10, batch_size=64, validation_data=(test_ped1_dataset, test_ped1_dataset))
    autoencoder_ped1_history = autoencoder_ped1.fit(train_ped1_dataset, train_ped1_dataset, epochs=10, batch_size=200, validation_data=(test_ped1_dataset, test_ped1_dataset))
    #Plotting loss
    #loss_plot(autoencoder_ped1_history)

    reconstruct_ped1 = autoencoder_ped1.predict(train_ped1_dataset)
    #print(reconstruct_ped1.shape)

    #Compute the reconstruction error of the training set and choose threshold
    train_loss_ped1 = mean_squared_error(reconstruct_ped1, train_ped1_dataset)
    #print(train_loss_ped1.shape)
    threshold_ped1 = np.mean(train_loss_ped1) + np.std(train_loss_ped1)
    print("Threshold for ped1: ", threshold_ped1)
    #Plotting
    #plt.hist(train_loss_ped1, bins=50)
    #plt.xlabel("Train loss")
    #plt.ylabel("No of examples")
    #plt.show()
    preds = predict(autoencoder_ped1, test_ped1_dataset, threshold_ped1)
    print_stats(preds, test_ped1_label)


if __name__ == '__main__':
    main()
