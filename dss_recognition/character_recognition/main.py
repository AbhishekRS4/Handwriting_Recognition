import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
%matplotlib inline
import train
import model
import train

pretrain = True
batch_size= 32
input_size = (64, 64)
optimizer = 'adam'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


path_train_data ="char_recog/train"
path_test_data ="char_recog/val"

import splitfolders

splitfolders.ratio('char_training', output="char_recog", seed=1337, ratio=(0.8, 0.2))

subfolder =["train", "val"]
for sf in subfolder:
    mainfolder= f"char_recog/{sf}"
    folders = os.listdir(mainfolder)
    for folder in  folders:
        files = os.listdir(f"{mainfolder}/{folder}")
        for f in files:
            impath = f"{mainfolder}/{folder}/{f}"
            newfile = "".join(f.split(".")[0:-1])
            newpath = f"{mainfolder}/{folder}/{newfile}.png"
            img = Image.open(impath).convert("RGBA")
            img.save(newpath)


def char_recog():
    model= model.alexnet(27, (64,64,3),"adam")
    print("Now loading pretrained weights.....")
    model.load_weights("f"{name}.h5".h5")

if __name__ == '__main__':
    input_shape = (64, 64, 3)
    optimizer = 'adam'
    # (1) Create a model
    model= model.alexnet(27, (64,64,3),"adam")
    # (2) Compile
    #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])
    

    # (3) Train
    if pretrain ==True: 
        print("Pretraining")
        pre_history = train.training("alexnet", (64, 64), 32)
        print("pretrain done")
    history = train.training(model, "alexnet", (64, 64), 32)
    
    # (4) Print results
    if pretrain: 
        print(pre_history.history) 
    
    print(history.history)
    plt.plot(history.history['accuracy'])
    plt.title('Alexnet')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
