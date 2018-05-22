# -*- coding: utf-8 -*-
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize
import sklearn


def catelab(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]] = 1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x


# generator that we will use to read the data from the directory
def data_gen_small(img_dir, mask_dir, batch_size, dims, n_labels, size):
    img_list = []
    label_list = []
    for i in range(size):
        img_list.append(img_dir + str(i) + ".png")
        label_list.append(mask_dir + str(i) + '.png')

    while True:
        for offset in range(0, size, batch_size):
            imgs = []
            labels = []
            batch_samples_img = img_list[offset:offset + batch_size]
            batch_samples_label = label_list[offset:offset + batch_size]

            for batch_sample in batch_samples_img:
                # images
                print(batch_sample)
                original_img = cv2.imread(batch_sample)[:, :, ::-1]
                # original_img = cv2.imread(img_dir + str(i) +".png")
                # resized_img = cv2.resize(original_img, (dims[0], dims[1],3))
                array_img = img_to_array(original_img)
                imgs.append(array_img)
                # imgs.append(original_img)

            for batch_sample in batch_samples_label:
                # masks
                original_mask = cv2.imread(batch_sample)
                # resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
                array_mask = catelab(original_mask[:, :, 0], dims, n_labels)
                labels.append(array_mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield sklearn.utils.shuffle(imgs, labels)
    # if len(imgs) == batch_size:
    #   yield imgs, labels
    #  imgs=[]
    #  labels=[]


def data_gen_small_val(img_dir, mask_dir, batch_size, dims, n_labels, size):
    # size=size+800
    img_list = []
    label_list = []
    for i in range(size):
        img_list.append(img_dir + str(i + 800) + ".png")
        label_list.append(mask_dir + str(i + 800) + '.png')
    while True:
        for offset in range(0, size, batch_size):
            imgs = []
            labels = []
            batch_samples_img = img_list[offset:offset + batch_size]
            batch_samples_label = label_list[offset:offset + batch_size]

            for batch_sample in batch_samples_img:
                # images
                print(batch_sample)
                original_img = cv2.imread(batch_sample)[:, :, ::-1]
                # original_img = cv2.imread(img_dir + str(i) +".png")
                # resized_img = cv2.resize(original_img, (dims[0], dims[1],3))
                array_img = img_to_array(original_img)
                imgs.append(array_img)
                # imgs.append(original_img)

            for batch_sample in batch_samples_label:
                # masks
                original_mask = cv2.imread(batch_sample)
                # resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
                array_mask = catelab(original_mask[:, :, 0], dims, n_labels)
                labels.append(array_mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield sklearn.utils.shuffle(imgs, labels)


