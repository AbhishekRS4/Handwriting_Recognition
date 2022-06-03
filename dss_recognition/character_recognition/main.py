import os
import tensorflow as tf
import train
import splitfolders
import load_alexnet
from PIL import Image


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

path_monkbrill='char_training'
expected_path='char_recog'

def splitting_folders(path,path_1):
    splitfolders.ratio(path, output=path_1, seed=1337, ratio=(0.8, 0.2))
    subfolder = ["train", "val"]
    for sf in subfolder:
        mainfolder = f"{path_1}/{sf}"
        folders = os.listdir(mainfolder)
        for folder in folders:
            files = os.listdir(f"{mainfolder}/{folder}")
            for f in files:
                impath = f"{mainfolder}/{folder}/{f}"
                newfile = "".join(f.split(".")[0:-1])
                newpath = f"{mainfolder}/{folder}/{newfile}.png"
                img = Image.open(impath).convert("RGBA")
                img.save(newpath)
    return mainfolder

splitting_folders(path_monkbrill,expected_path)
model = load_alexnet.model(load_weights=False)
train.training("alexnet",model)
