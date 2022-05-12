import os
import sys
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model_crnn import CRNN
from logger_utils import CSVWriter, write_json_file
from dataset import HWRecogIAMDataset, split_dataset, get_dataloaders_for_training


def train(hw_model, optimizer, criterion, train_loader, device):
    hw_model.train()
    train_running_loss = 0.0
    num_train_samples = len(train_loader.dataset)
    num_train_batches = len(train_loader)

    for images, labels, length_labels in train_loader:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        length_labels = length_labels.to(device, torch.long)

        batch_size = images.size(0)
        optimizer.zero_grad()
        logits = hw_model(images)
        log_probs = F.log_softmax(logits, dim=2)

        length_outputs = torch.LongTensor([logits.size(0)] * batch_size)
        length_labels = torch.flatten(length_labels)

        loss = criterion(log_probs, labels, length_outputs, length_labels)
        train_running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hw_model.parameters(), 5) # gradient clipping with 5
        optimizer.step()

    train_loss = train_running_loss / num_train_batches
    return train_loss

def validate(hw_model, criterion, valid_loader, device):
    hw_model.eval()
    valid_running_loss = 0.0
    num_valid_samples = len(valid_loader.dataset)
    num_valid_batches = len(valid_loader)

    with torch.no_grad():
        for images, labels, length_labels in valid_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            length_labels = length_labels.to(device, torch.long)

            batch_size = images.size(0)
            logits = hw_model(images)
            log_probs = F.log_softmax(logits, dim=2)
            length_outputs = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, labels, length_outputs, length_labels)
            valid_running_loss += loss.item()

        valid_loss = valid_running_loss / num_valid_batches
    return valid_loss

def train_hw_recognizer(FLAGS):
    file_txt_labels = os.path.join(FLAGS.dir_dataset, "iam_lines_gt.txt")
    dir_images = os.path.join(FLAGS.dir_dataset, "img")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA device not found, so exiting....")
        sys.exit(0)

    train_x, valid_x, train_y, valid_y = split_dataset(file_txt_labels, for_train=True)
    num_train_samples = len(train_x)
    num_valid_samples = len(valid_x)
    train_loader, valid_loader = get_dataloaders_for_training(
        train_x, train_y, valid_x, valid_y,
        dir_images=dir_images, image_height=FLAGS.image_height, image_width=FLAGS.image_width,
        batch_size=FLAGS.batch_size,
    )

    dir_model = f"model_{FLAGS.which_hw_model}"
    if not os.path.isdir(dir_model):
        print(f"creating directory: {dir_model}")
        os.makedirs(dir_model)

    file_logger_train = os.path.join(dir_model, "train_metrics.csv")
    csv_writer = CSVWriter(
        file_name=file_logger_train,
        column_names=["epoch", "loss_train", "loss_valid"]
    )

    file_params = os.path.join(dir_model, "params.json")
    write_json_file(file_params, vars(FLAGS))

    num_classes = len(HWRecogIAMDataset.LABEL_2_CHAR) + 1
    print(f"task - handwriting recognition")
    print(f"model: {FLAGS.which_hw_model}")
    print(f"learning rate: {FLAGS.learning_rate:.6f}, weight decay: {FLAGS.weight_decay:.6f}")
    print(f"batch size : {FLAGS.batch_size}, image height: {FLAGS.image_height}, image width: {FLAGS.image_width}")
    print(f"num train samples: {num_train_samples}, num validation samples: {num_valid_samples}\n")
    hw_model = CRNN(num_classes, FLAGS.image_height)
    hw_model.to(device)

    optimizer = torch.optim.Adam(hw_model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    criterion = nn.CTCLoss(reduction="mean", zero_infinity=True)

    print(f"training of handwriting recognition model {FLAGS.which_hw_model} started\n")
    for epoch in range(1, FLAGS.num_epochs+1):
        time_start = time.time()
        train_loss = train(hw_model, optimizer, criterion, train_loader, device)
        valid_loss = validate(hw_model, criterion, valid_loader, device)
        time_end = time.time()
        print(f"epoch: {epoch}/{FLAGS.num_epochs}, time: {time_end-time_start:.3f} sec.")
        print(f"train loss: {train_loss:.6f}, validation loss: {valid_loss:.6f}\n")

        csv_writer.write_row(
            [
                epoch,
                round(train_loss, 6),
                round(valid_loss, 6),
            ]
        )
        torch.save(hw_model.state_dict(), os.path.join(dir_model, f"{FLAGS.which_hw_model}_{epoch}.pth"))
    print(f"Training of handwriting recognition model {FLAGS.which_hw_model} complete!!!!")
    csv_writer.close()
    return

def main():
    learning_rate = 3e-5
    weight_decay = 1e-6
    batch_size = 64
    num_epochs = 50
    image_height = 32
    image_width = 768
    which_hw_model = "crnn"
    dir_dataset = "/home/abhishek/Desktop/RUG/hw_recognition/IAM-data/"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate to use for training")
    parser.add_argument("--weight_decay", default=weight_decay,
        type=float, help="weight decay to use for training")
    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="batch size to use for training")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="num epochs to train the model")
    parser.add_argument("--image_height", default=image_height,
        type=int, help="image height to be used to train the model")
    parser.add_argument("--image_width", default=image_width,
        type=int, help="image width to be used to train the model")
    parser.add_argument("--dir_dataset", default=dir_dataset,
        type=str, help="full directory path to the dataset")
    parser.add_argument("--which_hw_model", default=which_hw_model,
        type=str, choices=["crnn"], help="which model to train")

    FLAGS, unparsed = parser.parse_known_args()
    train_hw_recognizer(FLAGS)
    return

if __name__ == "__main__":
    main()
