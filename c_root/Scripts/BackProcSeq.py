# import
import cv2 as cv
import numpy as np
import math
import os
from tqdm import tqdm

"""
Author:     Marc Pitteloud
Date:       June 9, 2023
Description: This script adds background and noise to the tumble motion images.
"""

## Global variables
g_p: float = 0.05
r_p: float = 0.05
sp_p: float = 0.05
s_p: float = 0.05

# Paths
input_dir   = os.path.abspath("input")
output_dir  = os.path.abspath("output")


"""rgb_dir     = os.path.abspath(os.path.join(input_dir,"Tumble",motion_id,rgb))
back_dir    = os.path.abspath( os.path.join(input_dir,"Tumble",motion_id,back))
backimg_dir = os.path.abspath(os.path.join("earthPhotos","rotated"))
noise_dir   = os.path.abspath(os.path.join(input_dir,"Tumble",motion_id,noise))
mask_dir    = os.path.abspath(os.path.join(input_dir,"Tumble",motion_id mask))"""



rgb_dir: str = 'C:\\Users\\marcp\\Documents\\BachProj\\Tumble\\02003\\rgb'
back_dir: str = 'C:\\Users\\marcp\\Documents\\BachProj\\Tumble\\02003\\back'
backimg_dir: str = 'C:\\Users\\marcp\\Documents\\BachProj\\Tumble\\02003\\back_seq'
noise_dir: str = 'C:\\Users\\marcp\\Documents\\BachProj\\Tumble\\02003\\noise'
mask_dir: str = 'C:\\Users\\marcp\\Documents\\BachProj\\Tumble\\02003\\mask'
earth_img: str = 'C:\\Users\\marcp\\Documents\\BachProj\\Earth_7.png'

# Do you want to use crops of the earth_img or did you already put images in backimg_dir?
crop: bool = True

# Do you want to add background to the images or just noise?
with_background: bool = True


def main():
    if with_background:
        background(crop)
        print('Background added')

    print('Adding noise')
    addNoise(with_background)
    print('Noise added')


########################################################################################################################
# Background functions
def background(crop: bool):
    if crop:
        print('Cropping images')
        cropImage_seq(backimg_dir, cv.imread(earth_img), (1024, 1024), len(os.listdir(rgb_dir)))

    print('Adding background')
    addBackground_seq(rgb_dir, mask_dir, backimg_dir, back_dir)


def getIndices() -> list[tuple[int, int]]:
    indices = []
    for i in range(-10, 10 + 1):
        for j in range(-10, 10 + 1):
            if 25 <= i ** 2 + j ** 2 <= 100:
                indices.append((i, j))
    return indices


# Method to crop for a sequence of images
def cropImage_seq(output_directory: str, img: np.ndarray, gnr_image_size: tuple[int, int], n: int = 200):
    # Get the size of the image
    height, width, channels = img.shape
    tries: int = 0
    while True:
        start_x: int = np.random.randint(0, width)
        start_y: int = np.random.randint(0, height)
        indices: list[tuple[int, int]] = getIndices()
        print(indices)
        motion: tuple[int, int] = indices[np.random.randint(0, len(indices))]
        print(f"x: {start_x}, y: {start_y}, motion: {motion}, tries: {tries}")
        tries += 1
        if 0 <= start_x + gnr_image_size[0] < height \
                and 0 <= start_x + gnr_image_size[0] + motion[0] * n < height \
                and 0 <= start_y + gnr_image_size[1] < width \
                and 0 <= start_y + gnr_image_size[1] + motion[1] * n < width \
                and 0 <= start_x < height \
                and 0 <= start_x + motion[0] * n < height \
                and 0 <= start_y < width \
                and 0 <= start_y + motion[1] * n < width:
            break
    for i in tqdm(range(n)):
        x = start_x + motion[0] * i
        y = start_y + motion[1] * i
        cropped: np.ndarray = img[x:x + gnr_image_size[0], y:y + gnr_image_size[1]]
        imgName: str = f'{i:04d}.png'
        curr_output_directory = os.path.join(output_directory, imgName)
        cv.imwrite(curr_output_directory, cropped)


def addBackground_seq(rgb_path, mask_path, seq_path, output_directory):
    rgbList: list[str] = os.listdir(rgb_path)

    for imgName in tqdm(rgbList):
        img_path: str = os.path.join(rgb_path, imgName)
        img: np.ndarray = np.array(cv.imread(img_path))

        # Read the mask
        mask_path_2: str = os.path.join(mask_path, imgName)
        mask: np.ndarray = np.array(cv.imread(mask_path_2))

        # Get the background
        bg_path: str = os.path.join(seq_path, imgName)
        bg: np.ndarray = np.array(cv.imread(bg_path))

        # Add the background to the image
        img[mask != 0] = bg[mask != 0]

        # Save the image
        output_path: str = os.path.join(output_directory, imgName)
        cv.imwrite(output_path, img)


########################################################################################################################
# Noise functions
# Methode that adds noise to the noise_directory
def addNoise(with_s: bool = False):
    # To create the .txt file with the type of noise
    type_of_noise: list[str] = []
    n: int = 0

    input_img: list[str] = os.listdir(back_dir)
    for imgName in tqdm(input_img):
        n += 1

        # Read the image
        imgPath: str = os.path.join(back_dir, imgName)
        img: np.ndarray = np.array(cv.imread(imgPath))

        # Output path
        outPath: str = os.path.join(noise_dir, imgName)

        # Add the noise and saves the image in output
        noise: int = np.random.randint(0, 100)
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
            if with_s:
                cv.imwrite(outPath, img)
                type_of_noise.append('n')
            else:
                # Read the mask
                maskPath: str = os.path.join(mask_dir, imgName)
                mask: np.ndarray = np.array(cv.imread(maskPath))

                cv.imwrite(outPath, starry_noise(img, mask))
                type_of_noise.append('s')
        else:
            cv.imwrite(outPath, img)
            type_of_noise.append('n')
    write_type_of_noise(type_of_noise)


def read_first_row():
    img_names: list[str] = []
    with open('back.txt', 'r') as file:
        for i, line in enumerate(file):
            line: str = line.strip()
            if line:
                number: str = line.split(' ')[0]
                img_names.append(number)
    return np.array(img_names)


def write_type_of_noise(type_of_noise):
    with open('noise.txt', 'w') as file:
        for i in range(len(type_of_noise)):
            line: str = f'{i:04d}.png {type_of_noise[i]}\n'
            file.write(line)


def white_gaussian_noise(img, mean=0, sigma=5 ** (1 / 2)):
    noise = np.random.normal(mean, sigma, img.shape)

    # Add the noise to the input image
    noisy_img = img + noise

    # Clamping between 0 and 255 and as type uint8
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = np.round(noisy_img).astype(np.uint8)

    # Returns the output image
    return noisy_img


def rayleigh_fading_noise(img, scale=3.4):
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
    radius = np.random.randint(1, 5)
    for i in range(radius):
        for j in range(radius):
            if math.sqrt(i ** 2 + j ** 2) < radius:
                if x + i < img.shape[0] and y + j < img.shape[1]:
                    img[x + i, y + j] = c


def starry_noise(img, mask):
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
