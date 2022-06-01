# Handwriting Recognition
***
* This repository contains all code for the three tasks of the [Handwriting Recognition](https://www.rug.nl/ocasys/fwn/vak/show?code=WMAI019-05), Master's course of the University of Groningen.

## Team members
* Jeroen
* Milan
* Manuel
* Abhishek

## Dead Sea Scrolls
### Matlab engine Setup
* For the line segmentation, two approaches are used: a histogram method and a blob-line method. The latter uses the
implementation by Kurar Barakat et al.[[1]](#1) in Matlab. For this to function correctly, the user must have the MATLAB Engine
installed. This enables MATLAB code to be run directly from a Python script. The MATLAB engine can be installed as follows:

#### Verify installation
* Before you install, verify your Python and MATLAB configurations.
* Check that your system has a supported version of Python and MATLAB R2014b or later. Version R2021b was used in our case.
This version supports Python 3.7, 3.8 and 3.9.
* Next, find the path to the MATLAB folder. Start MATLAB and type `matlabroot` in the command window. Copy the path returned
by `matlabroot`.

#### Windows
Open a command prompt and type the following commands:
```
cd "MATLABROOT\extern\engines\python"
python setup.py install
```

#### Mac/Linux
```
cd "MATLABROOT/extern/engines/python"
python setup.py install
```

### Python Package dependencies
* The python package dependencies can be found in [requirements.txt](requirements.txt)

### Running
* To run the full pipeline for the line segmentation, character segmentation and character recognition, first ensure the
dataset is located in the correct folder, using the correct naming scheme.

## IAM Dataset
### Setup
* The python package dependencies can be found in [requirements.txt](requirements.txt)

### Running the final trained model to generate predictions
* The predictions of the model will be saved in individual txt files i.e. one for each image, in a directory named `results`.
The script [line_recognition/final_evaluation.py](line_recognition/final_evaluation.py) can be run in the following way.
```
python3 final_evaluation.py --dir_images <path_to_dir> --file_model <path_to_model_file>
```

## References
<a id="1">[1]</a>
Berat Kurar, B., Cohen, R., Droby, A., Rabaev, I. & El-Sana, J. (2020). Learning-Free Text Line Segmentation for Historical
Handwritten Documents.
Applied Sciences, 10, 8276; doi:10.3390/app10228276.
[paper link](https://www.researchgate.net/profile/Berat-Barakat/publication/347109911_Learning-Free_Text_Line_Segmentation_for_Historical_Handwritten_Documents/links/6005e26a45851553a053b11c/Learning-Free-Text-Line-Segmentation-for-Historical-Handwritten-Documents.pdf)

<a id="2">[2]</a>
Dutta, Kartik & Krishnan, Praveen & Mathew, Minesh & Jawahar, C.V.. (2018). Improving CNN-RNN Hybrid Networks for Handwriting Recognition.
80-85, doi:10.1109/ICFHR-2018.2018.00023.
[paper link](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2018/improving-cnn-rnn.pdf)
