import argparse
import numpy as np

from dataset import read_IAM_label_txt_file

def list_unique_characters_in_IAM_dataset(FLAGS):
    _, all_labels = read_IAM_label_txt_file(FLAGS.file_txt_labels)

    num_labels = len(all_labels)
    print(f"num labels : {num_labels}")
    unique_chars = []

    for label in all_labels:
        unique_chars = unique_chars + list(np.unique(np.array(list(label))))

    unique_chars = sorted(unique_chars)
    unique_chars = np.array(unique_chars)
    unique_chars = np.unique(unique_chars)
    unique_chars = ''.join(unique_chars)
    
    # prints all unique chars in the IAM dataset
    print(unique_chars)

    # prints the number of unique chars in the IAM dataset
    print(f"Number of unique characters : {len(unique_chars)}")
    return

def main():
    file_txt_labels = "/home/abhishek/Desktop/RUG/hw_recognition/IAM-data/iam_lines_gt.txt"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_txt_labels", default=file_txt_labels,
        type=str, help="full path to label text file")

    FLAGS, unparsed = parser.parse_known_args()
    list_unique_characters_in_IAM_dataset(FLAGS)
    return

if __name__ == "__main__":
    main()
