import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import os

import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np

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

hebrew_characters = {
    0:'alef',
    1: 'ayin',
    2: 'bet',
    3: 'dalet',
    4: 'gimel',
    5: 'he',
    6: 'het',
    7: 'kaf',
    8: 'kaf-final',
    9: 'lamed',
    10: 'mem',
    11: 'mem-medial',
    12: 'nun-final',
    13: 'nun-medial',
    14: 'pe',
    15: 'pe-final',
    16: 'qof',
    17: 'resh',
    18: 'samekh',
    19: 'shin',
    20: 'taw',
    21: 'tet',
    22: 'tsadi-final',
    23: 'tsadi-medial',
    24: 'waw',
    25: 'yod',
    26: 'zayin'
    }


expected_width = 64
expected_height = 64
'''
image = cv2.imread("1.png")
# Preprocessing the image
image = cv2.resize(image, (expected_width, expected_height), cv2.INTER_LINEAR)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
prediction_char = model.predict(image)
position_char = np.argmax(prediction_char)
print(hebrew_characters[position_char])

'''
segmented_characters_path = 'testing_model'
for subdir, dirs, files in os.walk(segmented_characters_path):
        predicted_char = []
        i = 0
        for f in files:
            original_image = cv2.imread(f"{subdir}/{f}")
            image = cv2.resize(original_image, (expected_width, expected_height), cv2.INTER_LINEAR)
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            prediction_char = model.predict(image)
            position_char = np.argmax(prediction_char)
            print(hebrew_characters[position_char])
            #if Viterbi > x:
            #prob.append(hebrew_characters) #np.argmax
            i = i + 1
        #need to reverse, since  hebrew characters are read from right to left
        #predicted_char.reverse()


