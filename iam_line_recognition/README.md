# Task - IAM dataset line recognition
***

### Code Inspiration
* Some code is inspired from [https://github.com/GitYCC/crnn-pytorch](https://github.com/GitYCC/crnn-pytorch) and [https://github.com/kris314/deep-text-recognition-benchmark](https://github.com/kris314/deep-text-recognition-benchmark)

### Setup
The python package dependencies can be found in [requirements.txt](requirements.txt)

### To train the model
* To train the model run the following
```
python3 train.py
```
* To list all the commandline arguments, run the following
```
python3 train.py --help
```

### Trained models
* The trained models can be downloaded from [here](https://drive.google.com/drive/folders/1c-aNgqMDB0xYfyKXmcFNKcrN-ldd0UvO?usp=sharing)

### Running the final trained model to generate predictions
* The script [line_recognition/final_iam_line_recognizer.py](line_recognition/final_iam_line_recognizer.py) can be run in the following way. Use `--which_hw_model` option to specify the model to be used. To run CRNN use `--which_hw_model crnn` and to run STN-CRNN use `--which_hw_model stn_crnn`
* The predictions of the model will be saved in individual .txt files i.e. one for each image, in a directory named `results_crnn` with CRNN model and `results_stn_crnn` with STN-CRNN model
```
python3 final_iam_line_recognizer.py --dir_images <full_path_to_dir> --which_hw_model <hw_model> --file_model <full_path_to_model_file>
```

## References
<a id="1">[1]</a>
Dutta, Kartik & Krishnan, Praveen & Mathew, Minesh & Jawahar, C.V.. (2018). Improving CNN-RNN Hybrid Networks for Handwriting Recognition.
80-85, doi:10.1109/ICFHR-2018.2018.00023.
[paper link](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2018/improving-cnn-rnn.pdf)
