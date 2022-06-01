# Handwriting Recognition
***
This repository contains all code for the three tasks of the Handwriting Recognition course of the University of Groningen.
It was developed by NAMESNAMESNAMESNAMES

## Dead Sea Scrolls
### Setup
For the line segmentation, two approaches are used: a histogram method and a blob-line method. The latter uses the
implementation by Kurar Barakat et al.[[1]](#1) in Matlab. For this to function correctly, the user must have the MATLAB Engine
installed. This enables MATLAB code to be run directly from a Python script. The MATLAB engine can be installed as follows:

#### Verify installation
Before you install, verify your Python and MATLAB configurations.

Check that your system has a supported version of Python and MATLAB R2014b or later. Version R2021b was used in our case.
This version supports Python 3.7, 3.8 and 3.9.

Next, find the path to the MATLAB folder. Start MATLAB and type `matlabroot` in the command window. Copy the path returned
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

### Running
To run the full pipeline for the line segmentation, character segmentation and character recognition, first ensure the
dataset is located in the correct folder, using the correct naming scheme.

## IAM Dataset

## References
<a id="1">[1]</a>
Berat Kurar, B., Cohen, R., Droby, A., Rabaev, I. & El-Sana, J. (2020). Learning-Free Text Line Segmentation for Historical
Handwritten Documents. 
Applied Sciences, 10, 8276; doi:10.3390/app10228276.