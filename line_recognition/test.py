import os
import sys
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model_crnn import CRNN
from ctc_decoder import ctc_decode
from dataset import HWRecogIAMDataset, split_dataset, get_dataloader_for_testing


def test(hw_model, test_loader, device):
    hw_model.eval()
    num_test_samples = len(test_loader.dataset)
    num_test_batches = len(test_loader)

    count = 0

    with torch.no_grad():
        for images, labels, length_labels in test_loader:
            if count == 5:
                break
            images = images.to(device, dtype=torch.float)

            batch_size = images.size(0)
            logits = hw_model(images)
            log_probs = F.log_softmax(logits, dim=2)
            pred_label = ctc_decode(log_probs)#, HWRecogIAMDataset.LABEL_2_CHAR)
            print(labels.cpu().numpy().tolist(), pred_label)
            count += 1
    return

def train_hw_recognizer(FLAGS):
    file_txt_labels = os.path.join(FLAGS.dir_dataset, "iam_lines_gt.txt")
    dir_images = os.path.join(FLAGS.dir_dataset, "img")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA device not found, so exiting....")
        sys.exit(0)

    test_x, test_y = split_dataset(file_txt_labels, for_train=False)
    num_test_samples = len(test_x)
    test_loader = get_dataloader_for_testing(
        test_x, test_y,
        dir_images=dir_images, image_height=FLAGS.image_height, image_width=FLAGS.image_width,
    )

    num_classes = len(HWRecogIAMDataset.LABEL_2_CHAR) + 1
    print(f"task - handwriting recognition")
    print(f"model: {FLAGS.which_hw_model}")
    print(f"image height: {FLAGS.image_height}, image width: {FLAGS.image_width}")
    print(f"num test samples: {num_test_samples}")

    hw_model = CRNN(num_classes, FLAGS.image_height)
    hw_model.to(device)
    hw_model.load_state_dict(torch.load(FLAGS.file_model))

    print(f"testing of handwriting recognition model {FLAGS.which_hw_model} started")
    test(hw_model, test_loader, device)
    return

def main():
    image_height = 32
    image_width = 768
    which_hw_model = "crnn"
    dir_dataset = "/home/abhishek/Desktop/RUG/hw_recognition/IAM-data/"
    file_model = "temp.pth"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--image_height", default=image_height,
        type=int, help="image height to be used to train the model")
    parser.add_argument("--image_width", default=image_width,
        type=int, help="image width to be used to train the model")
    parser.add_argument("--dir_dataset", default=dir_dataset,
        type=str, help="full directory path to the dataset")
    parser.add_argument("--which_hw_model", default=which_hw_model,
        type=str, choices=["crnn"], help="which model to train")
    parser.add_argument("--file_model", default=file_model,
        type=str, help="full path to trained model file (.pth)")

    FLAGS, unparsed = parser.parse_known_args()
    train_hw_recognizer(FLAGS)
    return

if __name__ == "__main__":
    main()
