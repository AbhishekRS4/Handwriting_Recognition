import os
import sys
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model_main import CRNN, STN_CRNN
from utils import ctc_decode, compute_wer_and_cer_for_sample
from dataset import HWRecogIAMDataset, split_dataset, get_dataloader_for_testing


def test(hw_model, test_loader, device, list_test_files):
    """
    ---------
    Arguments
    ---------
    hw_model : object
        handwriting recognition model object
    test_loader : object
        dataset loader object
    device : str
        device to be used for running the evaluation
    list_test_files : list
        list of all the test files
    """
    hw_model.eval()
    num_test_samples = len(test_loader.dataset)
    num_test_batches = len(test_loader)

    count = 0
    list_test_cers, list_test_wers = [], []

    with torch.no_grad():
        for images, labels, length_labels in test_loader:
            count += 1
            images = images.to(device, dtype=torch.float)
            log_probs = hw_model(images)
            pred_labels = ctc_decode(log_probs)
            labels = labels.cpu().numpy().tolist()

            str_label = [HWRecogIAMDataset.LABEL_2_CHAR[i] for i in labels]
            str_label = "".join(str_label)
            str_pred = [HWRecogIAMDataset.LABEL_2_CHAR[i] for i in pred_labels[0]]
            str_pred = "".join(str_pred)

            cer_sample, wer_sample = compute_wer_and_cer_for_sample(str_pred, str_label)
            list_test_cers.append(cer_sample)
            list_test_wers.append(wer_sample)
            print(f"progress: {count}/{num_test_samples}, test file: {list_test_files[count-1]}")
            print(f"{str_label} - label")
            print(f"{str_pred} - prediction")
            print(f"cer: {cer_sample:.3f}, wer: {wer_sample:.3f}\n")
    list_test_cers = np.array(list_test_cers)
    list_test_wers = np.array(list_test_wers)
    mean_test_cer = np.mean(list_test_cers)
    mean_test_wer = np.mean(list_test_wers)
    print(f"test set - mean cer: {mean_test_cer:.3f}, mean wer: {mean_test_wer:.3f}\n")
    return

def test_hw_recognizer(FLAGS):
    file_txt_labels = os.path.join(FLAGS.dir_dataset, "iam_lines_gt.txt")
    dir_images = os.path.join(FLAGS.dir_dataset, "img")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # choose a device for testing
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get the internal test set files
    test_x, test_y = split_dataset(file_txt_labels, for_train=False)
    num_test_samples = len(test_x)
    # get the internal test set dataloader
    test_loader = get_dataloader_for_testing(
        test_x, test_y,
        dir_images=dir_images, image_height=FLAGS.image_height, image_width=FLAGS.image_width,
    )

    num_classes = len(HWRecogIAMDataset.LABEL_2_CHAR) + 1
    print(f"task - handwriting recognition")
    print(f"model: {FLAGS.which_hw_model}")
    print(f"image height: {FLAGS.image_height}, image width: {FLAGS.image_width}")
    print(f"num test samples: {num_test_samples}")

    # load the right model
    if FLAGS.which_hw_model == "crnn":
        hw_model = CRNN(num_classes, FLAGS.image_height)
    elif FLAGS.which_hw_model == "stn_crnn":
        hw_model = STN_CRNN(num_classes, FLAGS.image_height, FLAGS.image_width)
    else:
        print(f"unidentified option : {FLAGS.which_hw_model}")
        sys.exit(0)
    hw_model.to(device)
    hw_model.load_state_dict(torch.load(FLAGS.file_model))

    # start testing of the model on the internal set
    print(f"testing of handwriting recognition model {FLAGS.which_hw_model} started\n")
    test(hw_model, test_loader, device, test_x)
    print(f"testing handwriting recognition model completed!!!!")
    return

def main():
    image_height = 32
    image_width = 768
    which_hw_model = "crnn"
    dir_dataset = "/home/abhishek/Desktop/RUG/hw_recognition/IAM-data/"
    file_model = "model_crnn/crnn_H_32_W_768_E_177.pth"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--image_height", default=image_height,
        type=int, help="image height to be used to predict with the model")
    parser.add_argument("--image_width", default=image_width,
        type=int, help="image width to be used to predict with the model")
    parser.add_argument("--dir_dataset", default=dir_dataset,
        type=str, help="full directory path to the dataset")
    parser.add_argument("--which_hw_model", default=which_hw_model,
        type=str, choices=["crnn", "stn_crnn"], help="which model to be used for prediction")
    parser.add_argument("--file_model", default=file_model,
        type=str, help="full path to trained model file (.pth)")

    FLAGS, unparsed = parser.parse_known_args()
    test_hw_recognizer(FLAGS)
    return

if __name__ == "__main__":
    main()
