"""
Author:     Marc Pitteloud
Date:       June 9, 2023
Description: This script adds background and noise to the random orientation images.
"""

# import
import cv2 as cv
import numpy as np
import math
import os
from tqdm import tqdm

## Global variables
g_p = 0.05
r_p = 0.05
sp_p = 0.05
s_p = 0.05

#motion
motion_id = "02000_val"

# Paths
proj_dir : str = "C:\\Users\\marin\\Documents\\BA5\\ProjB\\hubble"
input_dir : str = os.path.join(proj_dir,"input")
output_dir : str = os.path.join(proj_dir,"output")

rgb_dir     = os.path.join(input_dir,motion_id,"rgb")
back_dir    = os.path.join(input_dir,motion_id,"back")
backimg_dir = os.path.join("earthPhotos","rotated")
noise_dir   = os.path.join(input_dir,motion_id,"noise")
mask_dir    = os.path.join(input_dir,motion_id," mask")
earthp_dir  = os.path.join(proj_dir,"earthPhotos")



start: int = 10500
end: int = 12000

def log_backrdm(mess):
    with open(os.path.join(proj_dir,"log","backrdm_log.txt"),"a") as log_file:
        log_file(f"{mess}\n")
def main():
    read_and_add_background(start, end)
    print('Background added')
    addNoise()
    print('Noise added')


########################################################################################################################
# Background functions
def read_and_add_background(start: int, end: int):
    log_backrdm("reading background")
    bg_tab = read_numbers_from_file(start, end)
    arr = np.arange(3000)

    # Shuffle the array randomly
    np.random.shuffle(arr)
    rgb_tab = arr[:len(bg_tab)]

    # sort rgb_tab
    rgb_tab = np.sort(rgb_tab)

    # Txt file with the numbers
    write_numbers_to_file(rgb_tab, bg_tab)

    # Add the background to the images
    addBackground(bg_tab, rgb_tab)


def write_numbers_to_file(rgb_tab, bg_tab):
    log_backrdm("write_nb")
    with open('back.txt', 'w') as file:
        for i in range(len(rgb_tab)):
            line = f'{rgb_tab[i]:04d}.png {bg_tab[i]:05d}.png\n'
            file.write(line)


def read_numbers_from_file(start, end):
    log_backrdm("rd_nb")
    numbers = []
    with open('shuffled_numbers.txt', 'r') as file:
        for i, line in enumerate(file):
            if i < start:
                continue
            if i >= end:
                break
            line = line.strip()
            if line:
                number = int(line.split('.')[0])
                numbers.append(number)
    return np.array(numbers)


# Makes multiples crops of the image with the same size  and saves them in cropped images
# Used to generate the background for the random part of the dataset
def cropImage(img, size, seq=False, shift=50, n=0, offset_x=0, offset_y=0):
    # Get the size of the image
    log_backrdm("crop")
    height, width, channels = img.shape

    # Get the number of crops
    if seq:
        numCropsVert = math.floor((height - 1024) / shift)
        numCropsHori = math.floor((width - 1024) / shift)
    else:
        numCropsVert = math.floor(height / size)
        numCropsHori = math.floor(width / size)

    # Crop the image
    for i in range(numCropsVert):
        for j in range(numCropsHori):
            # Get the coordinates of the crop
            if seq:
                x = i * shift + offset_x
                y = j * shift + offset_y
            else:
                x = i * size
                y = j * size
            print(n, x, y)
            # Crop the image
            crop = img[x:x + size, y:y + size]

            # Save the image
            if seq:
                cv.imwrite(
                    os.path.join(earthp_dir,"cropped") + '{:05d}'.format(n) + '.png', crop)
            else:
                cv.imwrite(os.path.join(earthp_dir,"cropped") + str(i) + str(j) + '.png',
                           crop)
            n += 1
    return n


# Take a folder of images and rotate them for times by 90 degrees and store them in rotated_images
# Nb is the number of the first image, i.e. the offset
def rotateImages(nb):
    log_backrdm("rotimg")
    # Loop through all the images in the folder
    cropped_images = os.path.join(earthPhotos,"cropped")
    input_dir = cropped_images
    output_dir = os.path.join(earthPhotos,"rotated")

    for imgName in cropped_images:
        imgPath = os.path.join(input_dir, imgName)
        img = cv.imread(imgPath)

        for i in range(4):
            rotated = np.rot90(img, i)
            output_path = os.path.join(output_dir, f'{nb:05d}.png')
            cv.imwrite(output_path, rotated)
            print(nb)
            nb += 1
    return nb


def crop_and_rot():
    log_backrdm("croprot")
    input_dir = os.path.join(earthp_dir,"input")
    n = 0
    for imgName in input_dir:
        print(imgName)
        img = np.array(cv.imread(os.path.join(earthp_dir,"input", imgName)))
        n = cropImage(img, 1024, True, 65, n)
    print(f'Cropped images done!')
    rotateImages(0)


# Method that add the background from rotated images to the rendered images using the mask
def addBackground(bg_tab, rgb_tab):
    log_backrdm("addBack")
    if len(bg_tab) != len(rgb_tab):
        print('The two arrays must have the same length')
        return

    for i in tqdm(range(len(bg_tab))):
        # Read the image
        imgName: str = f'{rgb_tab[i]:04d}.png'
        imgPath: str = os.path.join(rgb_dir, imgName)
        img: np.ndarray = np.array(cv.imread(imgPath))

        # Read the mask
        maskPath_2 = os.path.join(mask_dir, imgName)
        mask = np.array(cv.imread(maskPath_2))

        # Get the background
        backgroundPath: str = os.path.join(backimg_dir, f'{bg_tab[i]:05d}.png')
        background: np.ndarray = np.array(cv.imread(backgroundPath))

        # Add the background to the image
        img[mask != 0] = background[mask != 0]

        # Save the image
        curr_output_dir: str = os.path.join(back_dir, imgName)
        cv.imwrite(curr_output_dir, img)


########################################################################################################################
# Noise functions
# Methode that adds noise to the noise_directory
def addNoise():
    log_backrdm("addnoise")
    # To create the .txt file with the type of noise
    type_of_noise = []
    n: int = 0

    # Read the first row of the file to know which images have earth background
    img_names: np.ndarray = read_first_row()

    input_img = os.listdir(rgb_dir)
    for imgName in tqdm(input_img):
        n += 1
        has_earth: bool = np.any(img_names == imgName)

        # Read the image
        if has_earth:
            imgPath = os.path.join(back_dir, imgName)
            img = np.array(cv.imread(imgPath))
        else:
            imgPath = os.path.join(rgb_dir, imgName)
            img = np.array(cv.imread(imgPath))

        # Output path
        outPath = os.path.join(noise_dir, imgName)

        # Add the noise and saves the image in output
        noise = np.random.randint(0, 100)
        if noise < g_p * 100:
            cv.imwrite(outPath, white_gaussian_noise(img))
            type_of_noise.append('g')
        elif (g_p + r_p) * 100 > noise > g_p * 100:
            cv.imwrite(outPath, rayleigh_fading_noise(img))
            type_of_noise.append('r')
        elif (g_p + r_p + sp_p) * 100 > noise > (g_p + r_p) * 100:
            cv.imwrite(outPath, salt_and_pepper_noise(img))
            type_of_noise.append('sp')
        elif (g_p + r_p + sp_p + s_p) * 100 > noise > (g_p + r_p + sp_p) * 100:
            if has_earth:
                cv.imwrite(outPath, img)
                type_of_noise.append('n')
            else:
                # Read the mask
                maskPath = os.path.join(mask_dir, imgName)
                mask = np.array(cv.imread(maskPath))

                cv.imwrite(outPath, starry_noise(img, mask))
                type_of_noise.append('s')
        else:
            cv.imwrite(outPath, img)
            type_of_noise.append('n')
    write_type_of_noise(type_of_noise)


def read_first_row():
    log_backrdm("readFrow")
    img_names = []
    with open('back.txt', 'r') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if line:
                number = line.split(' ')[0]
                img_names.append(number)
    return np.array(img_names)


def write_type_of_noise(type_of_noise):
    log_backrdm("wtypenoise")
    with open('noise.txt', 'w') as file:
        for i in range(len(type_of_noise)):
            line: str = f'{i:04d}.png {type_of_noise[i]}\n'
            file.write(line)


def white_gaussian_noise(img, mean=0, sigma=5 ** (1 / 2)):
    log_backrdm("whiteGn")
    noise = np.random.normal(mean, sigma, img.shape)

    # Add the noise to the input image
    noisy_img = img + noise

    # Clamping between 0 and 255 and as type uint8
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = np.round(noisy_img).astype(np.uint8)

    # Returns the output image
    return noisy_img


def rayleigh_fading_noise(img, scale=3.4):
    log_backrdm("rayleighN")
    # Generate Rayleigh fading noise
    # Adjust the scale parameter to control the noise level
    noise = np.random.rayleigh(scale, img.shape)

    # Add the noise to the input image
    noisy_img = img + noise

    # Clamping between 0 and 255 and as type uint8
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = np.round(noisy_img).astype(np.uint8)

    # Returns the output image
    return noisy_img


def salt_and_pepper_noise(img, prob=0.4):
    log_backrdm("condiment")
    h, w, _ = img.shape
    noisy_image = np.copy(img)

    # Generate random noise mask
    mask = np.random.random((h, w))

    # Set pixels to 0 (black) for salt noise
    noisy_image[mask < prob / 2] = 0

    # Set pixels to 255 (white) for pepper noise
    noisy_image[mask > 1 - prob / 2] = 255

    return noisy_image


# Color a patch of pixels
def color_patch(img, x, y, c):
    log_backrdm("colorpatch")
    radius = np.random.randint(1, 5)
    for i in range(radius):
        for j in range(radius):
            if math.sqrt(i ** 2 + j ** 2) < radius:
                if x + i < img.shape[0] and y + j < img.shape[1]:
                    img[x + i, y + j] = c


def starry_noise(img, mask):
    log_backrdm("starrynoise")
    height, width, channels = img.shape
    background = np.zeros([height, width, channels])

    # Change color of 0.5% of the pixels using numpy
    for i in range(height):
        for j in range(width):
            if np.random.randint(0, 1000) < 5:
                c = np.random.randint(0, 7)
                if c == 0:
                    color_patch(background, i, j, [175, 201, 255])
                elif c == 1:
                    color_patch(background, i, j, [199, 216, 255])
                elif c == 2:
                    color_patch(background, i, j, [255, 244, 243])
                elif c == 3:
                    color_patch(background, i, j, [255, 229, 207])
                elif c == 4:
                    color_patch(background, i, j, [255, 217, 178])
                elif c == 5:
                    color_patch(background, i, j, [255, 199, 142])
                elif c == 6:
                    color_patch(background, i, j, [255, 166, 81])

    background[mask == 0] = 0
    img = img + background
    return img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
