# Task - Dead Sea Scrolls recognition
***

## Dead Sea Scrolls
**_NOTE:_** Unfortunately, the blob-line segmentation method used in this pipeline was only able to be implemented in Windows. However, the horizontal histogram projection method for line segmentation should work on all systems when run individually, but it is preferable to run on Windows since the main pipeline is tested on Windows.
Documentation on the method was not sufficient to find an appropriate adaptation to Linux and Mac. Our apologies for this inconvenience.

### Matlab engine Setup
For the line segmentation, two approaches are used: a histogram method and a blob-line method. The histogram approach uses a peak detection algorithm by Roman Ptak et al.[[1]](#1), that incorporates a variable threshold. The latter uses the
implementation by Kurar Barakat et al.[[2]](#2) in Matlab. For this to function correctly, the user must have the MATLAB Engine
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
python3 setup.py install
```

### Matlab Dependencies
* Image Processing Toolbox
* MATLAB Support for MinGW-w64 C/C++ Compiler
* MATLAB Compiler
* MATLAB Compiler SDK

### Python Package dependencies
The python package dependencies can be found in [requirements.txt](requirements.txt)

### Running the pipeline
To run the full pipeline for the line segmentation, character segmentation and character recognition, first ensure the
dataset is using the correct naming scheme. This means all binarized versions of files contain 'binarized' in the filename.
For example, a file can be named: `P123-Fg001-R-C01-R01-binarized.jpg`

For the pipeline to take the data, either deposite all image files in the `image-data` folder in the `dss_recognition` directory,
or specify the location as a command line argument. By default, the resulting text files of the model are saved to a `results` directory in the `dss_recognition` directory.
However, if the user prefers, a custom saving directory can also be used by specifying this in the command line as well. `--line_segment_method` is a flag to specify which line segmenter to use, it can either be the blob-line method ('blob') or horizontal histogram projection ('hhp').

Running the pipeline is done by running the following file from the `dss_recognition` directory:

```
python3 dss_recognizer.py --dir_images <path_to_dir> --dir_save_predictions <path_to_dir> --line_segment_method <method>
```

## References
<a id="1">[1]</a>
Ptak, R., Zygadlo, B., Unold, O. (2017). Projection-Based Text Line Segmentation with a Variable
Threshold.
International Journal of Applied Mathematics and Computer Science, 27, doi:10.1515/amcs-2017-0014.
[paper link](https://www.researchgate.net/publication/315887219_Projection-Based_Text_Line_Segmentation_with_a_Variable_Threshold/fulltext/5909a119aca272f658fc7c62/Projection-Based-Text-Line-Segmentation-with-a-Variable-Threshold.pdf)

<a id="2">[2]</a>
Berat Kurar, B., Cohen, R., Droby, A., Rabaev, I. & El-Sana, J. (2020). Learning-Free Text Line Segmentation for Historical
Handwritten Documents.
Applied Sciences, 10, 8276; doi:10.3390/app10228276.
[paper link](https://www.researchgate.net/profile/Berat-Barakat/publication/347109911_Learning-Free_Text_Line_Segmentation_for_Historical_Handwritten_Documents/links/6005e26a45851553a053b11c/Learning-Free-Text-Line-Segmentation-for-Historical-Handwritten-Documents.pdf)
