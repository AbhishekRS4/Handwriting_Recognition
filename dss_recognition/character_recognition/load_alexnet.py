from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np


def model(load_weights=True):
    
    #Extra metrics to measure accuracy of the model
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    #Image size
    image_shape = (64,64,3)
    np.random.seed(1000)
    model = Sequential()
    
    #First Convolutional layer
    model.add(Conv2D(filters=96, input_shape=image_shape, kernel_size=(11,11), strides=(4,4),padding='valid'))
    model.add(Activation('relu'))

    #Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    #Second Convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    #Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    #Third Convolutional layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    #Fourth Convolutional layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    #Fifth Convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    #Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

    #Flaten to later pass to a Fully connected layer
    model.add(Flatten())

    #First FC
    model.add(Dense(4096))
    model.add(Activation('relu'))

    #Add dropout to reduce overfit
    model.add(Dropout(0.5))

    #Second FC
    model.add(Dense(4096))
    model.add(Activation('relu'))

    #Add Dropout
    model.add(Dropout(0.5))

    #Softmax to obtain 27 classes
    model.add(Dense(27))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy", recall_m, precision_m, f1_m])
    if load_weights:
         model.load_weights('character_recognition/alexnet.h5')

    return model
