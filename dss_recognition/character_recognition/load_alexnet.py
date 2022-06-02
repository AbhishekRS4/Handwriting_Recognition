import argparse
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np



def model():
    image_shape = (64,64,3)
    np.random.seed(1000)
    model = Sequential()
    #First Convolutional layer
    model.add(Conv2D(filters=96, input_shape=image_shape, kernel_size=(11,11), strides=(4,4),padding='valid'))
    model.add(Activation('relu'))

    #Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    #Second Convolutional layer
    model.add(Conv2D(filters=96, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    #Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    #Third Convolutional layer
    model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    #Fourth Convolutional layer
    model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    #Fifth Convolutional layer
    model.add(Conv2D(filters=22, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    #Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

    #Passing it to a fully connected layer, here we do the flatten!
    model.add(Flatten())

    #First Fully Connected layer has 4096 neurons
    model.add(Dense(300, input_shape=(64*64*3,)))
    model.add(Activation('relu'))

    #Add dropout to prevent overfitting
    model.add(Dropout(0.5))

    #Second Fully Connected layer
    model.add(Dense(200))
    model.add(Activation('relu'))

    #Add Dropout
    model.add(Dropout(0.5))

    #Output layer
    model.add(Dense(27))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
    model.load_weights('alexnet.h5')

    return model
