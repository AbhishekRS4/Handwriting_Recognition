import os
import sys
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage.io import imread
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import HWRecogIAMDataset
from model_main import CRNN, STN_CRNN
from utils import ctc_decode, compute_wer_and_cer_for_sample


class DatasetFinalEval(HWRecogIAMDataset):
    """
    Dataset class for final evaluation - inherits main dataset class
    """
    def __init__(self, dir_images, image_height=32, image_width=768):
        """
        ---------
        Arguments
        ---------
        dir_images : str
            full path to directory containing images
        image_height : int
            image height (default: 32)
        image_width : int
            image width (default: 768)
        """
        self.dir_images = dir_images
        self.image_files = [f for f in os.listdir(self.dir_images) if f.endswith(".png")]
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_height, self.image_width), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file_name = self.image_files[idx]
        image_gray = imread(os.path.join(self.dir_images, image_file_name))
        image_3_channel = np.repeat(np.expand_dims(image_gray, -1), 3, -1)
        image_3_channel = self.transform(image_3_channel)
        return image_3_channel

def get_dataloader_for_evaluation(dir_images, image_height=32, image_width=768, batch_size=1):
    """
    ---------
    Arguments
    ---------
    dir_images : str
        full path to directory containing images
    image_height : int
        image height (default: 32)
    image_width : int
        image width (default: 768)
    batch_size : int
        batch size to use for final evaluation (default: 1)

    -------
    Returns
    -------
    test_loader : object
        dataset loader object for final evaluation
    """
    test_dataset = DatasetFinalEval(dir_images=dir_images, image_height=image_height, image_width=image_width)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return test_loader

def final_eval(hw_model, device, test_loader, file_txt_preds, dir_images):
    """
    ---------
    Arguments
    ---------
    hw_model : object
        handwriting recognition model object
    device : str
        device to be used for running the evaluation
    test_loader : object
        dataset loader object
    file_txt_preds : str
        full path to text file where results need to be stored
    dir_images : str
        full path to directory containing test images
    """
    hw_model.eval()
    count = 0
    num_test_samples = len(test_loader.dataset)
    list_test_files = os.listdir(dir_images)
    fh_preds = open(file_txt_preds, "w", encoding="utf-8", newline="\n")

    with torch.no_grad():
        for image_test in test_loader:
            file_test = list_test_files[count]
            count += 1
            """
            if count == 11:
                break
            """
            image_test = image_test.to(device, dtype=torch.float)

            log_probs = hw_model(image_test)
            pred_labels = ctc_decode(log_probs)
            str_pred = [DatasetFinalEval.LABEL_2_CHAR[i] for i in pred_labels[0]]
            str_pred = "".join(str_pred)

            fh_preds.write(file_test+"\n")
            fh_preds.write(str_pred+"\n\n")

            print(f"progress: {count}/{num_test_samples}, test file: {list_test_files[count-1]}")
            print(f"{str_pred}\n")
    fh_preds.close()
    return

def test_hw_recognizer(FLAGS):
    file_txt_preds = f"predictions_{FLAGS.which_hw_model}.txt"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    num_classes = len(DatasetFinalEval.LABEL_2_CHAR) + 1
    print(f"task - handwriting recognition")
    print(f"model: {FLAGS.which_hw_model}")
    print(f"image height: {FLAGS.image_height}, image width: {FLAGS.image_width}")

    if FLAGS.which_hw_model == "crnn":
        hw_model = CRNN(num_classes, FLAGS.image_height)
    elif FLAGS.which_hw_model == "stn_crnn":
        hw_model = STN_CRNN(num_classes, FLAGS.image_height, FLAGS.image_width)
    else:
        print(f"unidentified option : {FLAGS.which_hw_model}")
        sys.exit(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hw_model.to(device)
    hw_model.load_state_dict(torch.load(FLAGS.file_model))

    test_loader = get_dataloader_for_evaluation(
        dir_images=FLAGS.dir_images, image_height=FLAGS.image_height, image_width=FLAGS.image_width,
    )

    print(f"final evaluation of handwriting recognition model {FLAGS.which_hw_model} started\n")
    final_eval(hw_model, device, test_loader, file_txt_preds, FLAGS.dir_images)
    print(f"final evaluation of handwriting recognition model completed!!!!")
    return

def main():
    image_height = 32
    image_width = 768
    which_hw_model = "crnn"
    dir_images = "/home/abhishek/Desktop/RUG/hw_recognition/IAM-data/img/"
    file_model = "temp.pth"
    save_predictions = 1

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--image_height", default=image_height,
        type=int, help="image height to be used to train the model")
    parser.add_argument("--image_width", default=image_width,
        type=int, help="image width to be used to train the model")
    parser.add_argument("--dir_images", default=dir_images,
        type=str, help="full directory path to directory containing images")
    parser.add_argument("--which_hw_model", default=which_hw_model,
        type=str, choices=["crnn", "stn_crnn", "stn_pp_crnn"], help="which model to train")
    parser.add_argument("--file_model", default=file_model,
        type=str, help="full path to trained model file (.pth)")
    parser.add_argument("--save_predictions", default=save_predictions,
        type=int, choices=[0, 1], help="save or do not save the predictions (1 - save, 0 - do not save)")

    FLAGS, unparsed = parser.parse_known_args()
    test_hw_recognizer(FLAGS)
    return

if __name__ == "__main__":
    main()
