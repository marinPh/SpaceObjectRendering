"""
Author:     Marc Pitteloud
Date:       June 9, 2023
Description: This script crops and rotates the images in the input folder and saves them in the output folder.
"""

# organize imports

import numpy as np
import cv2 as cv
import os
import math
from tqdm import tqdm
from tqdm.contrib import itertools

# Global variables
input_dir   =  os.path.abspath("input")
crop_dir    = os.path.abspath(os.path.join(input_dir,earthPhotos,crop))
output_dir  = os.path.abspath("output")

rot_seq: bool = False
seq_len: int = 500

shift: int = 65


def main():
    if rot_seq:
        rotate_seq(seq_len)
    else:
        crop_rot()


def crop_rot():
    input_list = os.listdir(input_dir)
    n: int = 0
    for imgName in input_list:
        print(imgName)
        img: np.ndarray = cv.imread(os.path.join(input_dir, imgName))
        n = cropImage(img, 1024, 65, n)
    print(f'Cropped images done!')
    n = rotateImages(0)
    shuffleImages(n)
    print('All done!')


# Makes multiples crops of the image with the same size  and saves them in cropped images
def cropImage(img: np.ndarray, size: int, shift: int = 50, n: int = 0):
    # Get the size of the image
    height, width, channels = img.shape

    # Get the number of crops

    numCropsVert: int = math.floor((height - 1024) / shift)
    numCropsHori: int = math.floor((width - 1024) / shift)

    # Crop the image
    for i, j in itertools.product(range(numCropsVert), range(numCropsHori)):
        # Get the coordinates of the crop
        x = i * shift
        y = j * shift

        print(n, x, y)
        # Crop the image
        crop = img[x:x + size, y:y + size]

        # Save the image
        imgName = f'{n:05d}.png'
        cv.imwrite(os.path.join(crop_dir, imgName), crop)

        n += 1
    return n


# Take a folder of images and rotate them for times by 90 degrees and store them in rotated_images
# Nb is the number of the first image, i.e. the offset
def rotateImages(nb):
    # Loop through all the images in the folder
    cropped_images = os.listdir(crop_dir)
    for j in tqdm(range(len(cropped_images))):
        imgName = cropped_images[j]
        imgPath = os.path.join(input_dir, imgName)
        img = cv.imread(imgPath)

        for i in range(4):
            rotated = np.rot90(img, i)
            output_path = os.path.join(output_dir, f'{nb:05d}.png')
            cv.imwrite(output_path, rotated)
            nb += 1
    return nb


def rotate_seq(length):
    # Loop through all the images in the folder
    cropped_images = os.listdir(crop_dir)
    for j in tqdm(range(len(cropped_images))):
        imgName = cropped_images[j]
        imgPath = os.path.join(input_dir, imgName)
        img = cv.imread(imgPath)

        for i in range(4):
            rotated = np.rot90(img, i)
            output_path = os.path.join(output_dir, f'{j + 500 * i:04d}.png')
            cv.imwrite(output_path, rotated)


def shuffleImages(n: int):
    tab: np.ndarray = np.arange(n)
    tab = np.random.shuffle(tab)
    with open('shuffled_numbers.txt', 'w') as f:
        for i in tab:
            f.write(f'{i:05d}.png\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
