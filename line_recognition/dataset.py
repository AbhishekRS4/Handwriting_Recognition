import os
import torch
import torch.nn
import numpy as np
from PIL import Image
from skimage.io import imread
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def read_IAM_label_txt_file(file_txt_labels):
    label_file_handler = open(file_txt_labels, mode="r")
    all_lines = label_file_handler.readlines()
    num_lines = len(all_lines)

    all_image_files = []
    all_labels = []

    for cur_line_num in range(num_lines):
        if cur_line_num % 3 == 0:
            all_image_files.append(all_lines[cur_line_num].strip())
        elif cur_line_num %3 == 1:
            all_labels.append(all_lines[cur_line_num].strip())
        else:
            continue

    return all_image_files, all_labels

class HWRecogIAMDataset(Dataset):
    CHAR_SET = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    CHAR_2_LABEL = {char: i + 1 for i, char in enumerate(CHAR_SET)}
    LABEL_2_CHAR = {label: char for char, label in CHAR_2_LABEL.items()}

    def __init__(self, image_files, labels, dir_images, image_height=32, image_width=768, which_set="train"):
        self.labels = labels
        self.dir_images = dir_images
        self.image_files = image_files
        self.image_width = image_width
        self.image_height = image_height
        self.which_set = which_set

        if self.which_set == "train"
            # apply data augmentation only for train set
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_height, self.image_width), Image.BILINEAR),
                transforms.RandomAffine(degrees=[-0.75, 0.75], translate=[0, 0.05], scale=[0.75, 1],
                    shear=[-10, 15], interpolation=transforms.InterpolationMode.BILINEAR, fill=255,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
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

        label_string = self.labels[idx]
        label_encoded = [self.CHAR_2_LABEL[c] for c in label_string]
        label_length = [len(label_encoded)]

        label_encoded = torch.LongTensor(label_encoded)
        label_length = torch.LongTensor(label_length)

        return image_3_channel, label_encoded, label_length

def IAM_collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.cat(labels, 0)
    label_lengths = torch.cat(label_lengths, 0)
    return images, labels, label_lengths

def split_dataset(file_txt_labels, for_train=True):
    all_image_files, all_labels = read_IAM_label_txt_file(file_txt_labels)
    train_image_files, test_image_files, train_labels, test_labels = train_test_split(all_image_files, all_labels, test_size=0.1, random_state=4)
    train_image_files, valid_image_files, train_labels, valid_labels = train_test_split(train_image_files, train_labels, test_size=0.1, random_state=4)
    if for_train:
        return train_image_files, valid_image_files, train_labels, valid_labels
    else:
        return test_image_files, test_labels

def get_dataloaders_for_training(train_x, train_y, valid_x, valid_y, dir_images, image_height=32, image_width=768, batch_size=8):
    train_dataset = HWRecogIAMDataset(train_x, train_y, dir_images, image_height=image_height, image_width=image_width, which_set="train")
    valid_dataset = HWRecogIAMDataset(valid_x, valid_y, dir_images, image_height=image_height, image_width=image_width, which_set="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=IAM_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=IAM_collate_fn,
    )
    return train_loader, valid_loader

def get_dataloader_for_testing(test_x, test_y, dir_images, image_height=32, image_width=768, batch_size=1):
    test_dataset = HWRecogIAMDataset(test_x, test_y, dir_images=dir_images, image_height=image_height, image_width=image_width, which_set="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=IAM_collate_fn,
    )
    return test_loader
