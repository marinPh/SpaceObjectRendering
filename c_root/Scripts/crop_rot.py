"""
Author:     Marc Pitteloud
Date:       June 9, 2023
Description: This script crops and rotates the images in the input folder and saves them in the output folder.
"""

# organize imports
import sys
import numpy as np
import cv2 as cv
import os
import math
from tqdm import tqdm
from tqdm.contrib import itertools

# Global variables

# get image_name by command line 
#get index of "--"
index = sys.argv.index("--")



seq_len: int = int(sys.argv[index+1])#500

shift: int = int(sys.argv[index+2])#65

orientation_flag: str = sys.argv[index+3]#0



if sys.argv[index+ 4] == '':
    rot_seq: bool = False
else:
    rot_seq: bool = 1==int(sys.argv[index+4])# 0

    



parent_dir = os.path.dirname(os.path.dirname(__file__))
input_dir = os.path.join(parent_dir, "input", "earthImg")
crop_dir = os.path.join(parent_dir, "input", "backImg")
output_dir = os.path.join(parent_dir, "output", "backImg")




def main():
    earthImg = list(os.listdir(input_dir))
    print(f"images in folder : {earthImg}")
    print ("---->")
    if rot_seq:
        rotate_seq(seq_len)
    else:
        for i in earthImg:
            
            img = cv.imread(os.path.join(input_dir, i))

            cropImage(img, 1024, shift, 0, i)


def crop_rot():
    
    print(f"images in folder : {input_list}")
    print ("---->")
    print (f'Number of images: {len(input_list)}')
    n: int = 0
    
    for imgName in input_list:
        #print(imgName)

        img: np.ndarray = cv.imread(os.path.join(input_dir, imgName))
        n = cropImage(img, 1024, shift, n, imgName)
    print(f'Cropped images done!')
    n = rotateImages(0)
    shuffleImages(n)
    print('All done!')


# Makes multiples crops of the image with the same size  and saves them in cropped images
def cropImage(img: np.ndarray, size: int, shift: int = 50, n: int = 0, image_name: str = 'image'):
    image_name = image_name.split('.')[0]
    # Get the size of the image
    height, width, channels = img.shape

    # Get the number of crops

    numCropsVert: int = math.floor((height - 1024) / shift)
    print(f'Number of crops: {numCropsVert}')
    numCropsHori: int = math.floor((width - 1024) / shift)
    print(f'Number of crops: {numCropsHori}')

    # Crop the image
    for i, j in itertools.product(range(numCropsVert), range(numCropsHori)):
        # Get the coordinates of the crop
        x = i * shift
        y = j * shift

        #print(n, x, y)
        # Crop the image
        crop = img[x:x + size, y:y + size]

        # Save the image

        #if orientation_flag == -v then image image_index = j 
        #if orientation_flag == -h then image image_index = i
        #if orientation_flag == -vh then image image_index = i * numCropsHori + j

        if orientation_flag == '-v':
            imgName = f'{j:05d}.png'
            file_index = i
        elif orientation_flag == '-h':
            imgName = f'{i:05d}.png'
            file_index = j
        else:
            imgName = f'{i * numCropsHori + j:05d}.png'
            file_index = "all"
        

        
        new_output_dir = os.path.join(crop_dir,f"crop{file_index}_{image_name}")
        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)

        cv.imwrite(os.path.join( new_output_dir,imgName), crop)

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
            output_path = os.path.join(output_dir,"rotated", f'{nb:05d}.png')
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
    tab = np.shuffle(tab)
    with open('shuffled_numbers.txt', 'w') as f:
        for i in tab:
            f.write(f'{i:05d}.png\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
