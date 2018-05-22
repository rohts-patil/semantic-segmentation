# -*- coding: utf-8 -*-
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize


def catelab(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]]=1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x


# generator that we will use to read the data from the directory
def data_gen_small(img_dir, mask_dir,  batch_size, dims, n_labels,size):
        while True:
            imgs = []
            labels = []
            for i in range(size):
                # images
                original_img = cv2.imread(img_dir + str(i) +".png")[:, :, ::-1]
                #original_img = cv2.imread(img_dir + str(i) +".png")
                #resized_img = cv2.resize(original_img, (dims[0], dims[1],3))
                array_img = img_to_array(original_img)
                imgs.append(array_img)
                #imgs.append(original_img)
                # masks
                original_mask = cv2.imread(mask_dir + str(i)  + '.png')
                #resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
                array_mask = catelab(original_mask[:, :, 0], dims, n_labels)
                labels.append(array_mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels


def data_gen_small_val(img_dir, mask_dir,  batch_size, dims, n_labels,size):
        #size=size+800
        while True:
            imgs = []
            labels = []
            for i in range(size):
                # images
                original_img = cv2.imread(img_dir + str(i+800) +".png")[:, :, ::-1]
                #original_img = cv2.imread(img_dir + str(i) +".png")
                #resized_img = cv2.resize(original_img, (dims[0], dims[1],3))
                array_img = img_to_array(original_img)
                imgs.append(array_img)
                #imgs.append(original_img)
                # masks
                original_mask = cv2.imread(mask_dir + str(i+800)  + '.png')
                #resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
                array_mask = catelab(original_mask[:, :, 0], dims, n_labels)
                labels.append(array_mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels
