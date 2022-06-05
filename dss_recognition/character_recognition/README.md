# Handwriting Recognition Character training
***
This folder contains all code for the Character training for [Handwriting Recognition](https://www.rug.nl/ocasys/fwn/vak/show?code=WMAI019-05), Master's course of the University of Groningen.

The alexnet architecture was created based on http://dangminhthang.com/computer-vision/character-recognition-using-alexnet/ and the alexnet original paper https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf


* The model is located in "load_alexnet.py"
* The training and testing is located at "train.py"
* The alexnet.h5 are the weights saved after the training
* To train the alexnet run the file "main.py"
* The dataset used was monkbrill renamed to character_training.


### Running the final trained model to generate predictions
* The training was done according to alexnet architecture, and can be run with the command: 
The script [character_recognition/main.py](character_recognition/main.py) can be run in the following way.
```
python3 main.py>
```
