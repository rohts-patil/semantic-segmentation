import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D
from keras.optimizers import *
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers.merge import concatenate
from pooling import *

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply, Concatenate
from keras.utils import np_utils

class myUnet(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def preprocess_labels(label_image,obj):
		# Identify lane marking pixels (label is 6)
		lane_marking_pixels = (label_image[:,:,0] == ROAD_LINES).nonzero()
		road_pixels = (label_image[:,:,0] == ROADS).nonzero()
    	# Set lane marking pixels to road (label is 7)
		labels_new = np.zeros_like(label_image)
    	#labels_new = np.copy(label_image)
		labels_new[lane_marking_pixels] = ROADS
		if obj=='road' or obj=='both':
			labels_new[road_pixels] = ROADS
 
    	# Identify all vehicle pixels
		vehicle_pixels = (label_image[:,:,0] == VEHICLES).nonzero()
		if obj == 'car' or obj=='both':
			labels_new[vehicle_pixels] = VEHICLES
    	# Isolate vehicle pixels associated with the hood (y-position > 496)
		hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
		hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    	# Set hood pixel labels to 0
		labels_new[hood_pixels] = 0
    	# Return the preprocessed label image 
		return labels_new

	def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
		cw = (target.get_shape()[2] - refer.get_shape()[2]).value
		assert (cw >= 0)
		if cw % 2 != 0:
			cw1, cw2 = int(cw/2), int(cw/2) + 1
		else:
			cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
		ch = (target.get_shape()[1] - refer.get_shape()[1]).value
		assert (ch >= 0)
		if ch % 2 != 0:
			ch1, ch2 = int(ch/2), int(ch/2) + 1
		else:
			ch1, ch2 = int(ch/2), int(ch/2)

		return (ch1, ch2), (cw1, cw2)


	def get_unet(self):
		""" 
		inputs = Input((self.img_rows, self.img_cols,1))
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print("conv1 shape):",conv1.shape)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print("conv1 shape):",conv1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print("pool1 shape):",pool1.shape)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print("conv2 shape):",conv2.shape)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print("conv2 shape):",conv2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print("pool2 shape):",pool2.shape)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print("conv3 shape):",conv3.shape)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print("conv3 shape):",conv3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print("pool3 shape):",pool3.shape)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy']) 

		concat_axis = 3
		inputs = Input((self.img_rows, self.img_cols,3))
        #inputs = Input(shape = img_shape)

		conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
		conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
		conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
		conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
		conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
		conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

		up_conv5 = UpSampling2D(size=(2, 2))(conv5)
		ch, cw = self.self.get_crop_shape(conv4, up_conv5)
		crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
		up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
		conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
		conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

		up_conv6 = UpSampling2D(size=(2, 2))(conv6)
		ch, cw = self.self.get_crop_shape(conv3, up_conv6)
		crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
		up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis) 
		conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
		conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

		up_conv7 = UpSampling2D(size=(2, 2))(conv7)
		ch, cw = self.self.get_crop_shape(conv2, up_conv7)
		crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
		up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
		conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
		conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

		up_conv8 = UpSampling2D(size=(2, 2))(conv8)
		ch, cw = self.self.get_crop_shape(conv1, up_conv8)
		crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
		up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
		conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
		conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

		ch, cw = self.self.get_crop_shape(inputs, conv9)
		conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
		num_class =2
		conv10 = Conv2D(num_class, (1, 1))(conv9)

		model = models.Model(inputs=inputs, outputs=conv10)
		"""
		concat_axis = 3

		inputs = Input((self.img_rows, self.img_cols,3))
		conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)
		conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)	
		pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
		conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
		conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

		conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
		conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

		conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
		conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

		conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
		conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

		up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)
		ch, cw = self.get_crop_shape(conv4, up_conv5)
		crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
		up6   = concatenate([up_conv5, crop_conv4], axis=concat_axis)
		conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
		conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

		up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)
		ch, cw = self.get_crop_shape(conv3, up_conv6)
		crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)
		up7   = concatenate([up_conv6, crop_conv3], axis=concat_axis)
		conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
		conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

		up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)
		ch, cw = self.get_crop_shape(conv2, up_conv7)
		crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)
		up8   = concatenate([up_conv7, crop_conv2], axis=concat_axis)
		conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
		conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

		up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)
		ch, cw = self.get_crop_shape(conv1, up_conv8)
		crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)
		up9   = concatenate([up_conv8, crop_conv1], axis=concat_axis)
		conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
		conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)

    #ch, cw = self.get_crop_shape(inputs, conv9)
    #conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)
    #conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)
    
		flatten =  Flatten()(conv9)
		Dense1 = Dense(512, activation='relu')(flatten)
		BN =BatchNormalization() (Dense1)
		Dense2 = Dense(2, activation='sigmoid')(BN)
    
		model = Model(input=inputs, output=Dense2)
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

		return model

	


	def train(self,imgs_train, imgs_mask_train, imgs_test,):

		print("loading data")
		#imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print("loading data done")
		#model = self.get_unet()
		#model=CreateSegNet(np.shape(imgs_train[0]), 2)
		model=get_model()
		 
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(np.array(imgs_train), np.array(imgs_mask_train), batch_size=4, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

		return model
		#imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		#np.save('../results/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):

		print("array to image")
		imgs = np.load('imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("../results/%d.jpg"%(i))



def CreateSegNet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax"):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    segnet = Model(inputs=inputs, outputs=outputs, name="SegNet")
    segnet.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return segnet




def get_model():

    inputs = Input(shape=(600, 800, 3))

    conv_1 = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(act_1)
    act_2 = Activation('relu')(conv_2)

    deconv_1 = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same')(act_2)
    act_3 = Activation('relu')(deconv_1)

    merge_1 = concatenate([act_3, act_1], axis=3)

    deconv_2 = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same')(merge_1)
    act_4 = Activation('relu')(deconv_2)

    model = Model(inputs=[inputs], outputs=[act_4])

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])
    return model 


#if __name__ == '__main__':
#	myunet = myUnet()
#	myunet.train()
#	myunet.save_img()







