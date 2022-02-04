# script to copy a set number of first n files in a directory to a new directory
# usage: python3 break_dataset.py <directory> <number of files> <new directory>

import os
import sys
import shutil


def break_dataset(directory, number_of_files, new_directory):
    """
    Copy a set number of first n files in a directory to a new directory
    :param directory: directory to copy from
    :param number_of_files: number of files to copy
    :param new_directory: directory to copy to
    :return:
    """
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            if number_of_files > 0:
                shutil.copy(os.path.join(directory, file), os.path.join(new_directory, file))
                number_of_files -= 1
            else:
                break


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 break_dataset.py <directory> <number of files> <new directory>")
        sys.exit(1)
    break_dataset(sys.argv[1], int(sys.argv[2]), sys.argv[3])