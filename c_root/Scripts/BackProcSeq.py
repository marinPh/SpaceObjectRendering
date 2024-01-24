# import
import sys
import site
user_site_packages = site.getusersitepackages()
print (user_site_packages)
sys.path.append(user_site_packages)

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

#get object name from command line
object_name = sys.argv[sys.argv.index("--") + 1]
# get motion id from command line
motion_id = sys.argv[sys.argv.index("--") + 2]
#get noise or background flag from command line
noise_or_background = sys.argv[sys.argv.index("--") + 3]

# Object ID
object_id = object_name.split("_")[0]
print(f"motion_id: {motion_id}")



# Paths
rgb_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", f"{object_id}_{motion_id}")
back_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", f"{object_id}_{motion_id}","back")
backimg_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input", "backimg")
noise_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", f"{object_id}_{motion_id}","noise")
mask_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", f"{object_id}_{motion_id}","mask")
earth_img: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input", "earthImg")

if not os.path.exists(back_dir):
    os.makedirs(back_dir)
if not os.path.exists(noise_dir):
    os.makedirs(noise_dir)
if not os.path.exists(backimg_dir):
    os.makedirs(backimg_dir)
if not os.path.exists(earth_img):
    os.makedirs(earth_img)

# Do you want to use crops of the earth_img or did you already put images in backimg_dir?
# is backimg_dir empty?
crop: bool  = os.listdir(backimg_dir) == []
# Do you want to add background to the images or just noise?
with_background: bool =  noise_or_background == "-b" or noise_or_background == "-bu"   


def main():
    print ('into main')
    if with_background:
        background(crop)
        print('Background added')

    print('Adding noise')
    #addNoise(with_background)
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
def cropImage_seq(output_directory: str, img: np.ndarray, gnr_image_size: tuple[int, int], n: int = 100):
    # Get the size of the image
    height, width, channels = img.shape
    tries: int = 0
    while True:
        start_x: int = np.random.randint(0, width)
        start_y: int = np.random.randint(0, height)
        indices: list[tuple[int, int]] = getIndices()
        print(f"len indices: {len(indices)}")
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


def addBackground_seq(rgb_path, mask_path, back_dir, output_directory):
    print (f"rgb_path: {rgb_path}")
    print (f"mask_path: {mask_path}")
    print (f"back_dir: {back_dir}")
    print (f"output_directory: {output_directory}")

    rgbList: list[str] = os.listdir(rgb_path)
    rgbList = [x for x in rgbList if x.startswith("rgb") and not os.path.isdir(os.path.join(rgb_path, x))]


    #list the directories in back_dir that have at least len(rgbList) files
    backimgList: list[str] = os.listdir(back_dir)   
    backimgList = [x for x in backimgList if os.path.isdir(os.path.join(back_dir, x)) and len(os.listdir(os.path.join(back_dir, x))) >= len(rgbList)]

    
    #choose a random directory from backimgList 
    backimg_dir_2: str = os.path.join(backimg_dir, backimgList[np.random.randint(0, len(backimgList))])
    print(f"backimg_dir_2: {backimg_dir_2}")
     # keep only the files starting with rgb and no folders
    
    nb_back = len(os.listdir(backimg_dir_2))
    print (f"nb_back: {nb_back}")
    print (f"len(rgbList): {len(rgbList)}")
    if nb_back == len(rgbList):
        starting_index = 0
    else:
        starting_index = np.random.randint(0, nb_back - len(rgbList))
    print (f"starting_index: {starting_index}")
    #create 3 arrays of shape (len(rgbList), 1024, 1024, 3)
    rgb_list = []
    mask_list = []
    back_list = []
#tqdm is used to show a progress bar
    for i in tqdm(range(len(rgbList))):
        #each iteration append the rgb, mask and back image to the corresponding list
        rgb_list.append(cv.imread(os.path.join(rgb_path, rgbList[i])))
        mask_list.append(cv.imread(os.path.join(mask_path, rgbList[i][3:])))
        back_list.append(cv.imread(os.path.join(backimg_dir_2, f"{starting_index+i:04d}.png")))
    #create the numpy arrays of shape (len(rgbList), 1024, 1024, 3)
    rgb_array = np.array(rgb_list)
    mask_array = np.array(mask_list)
    back_array = np.array(back_list)
    print (f"rgb_array.shape: {rgb_array.shape}")
    print (f"mask_array.shape: {mask_array.shape}")
    print (f"back_array.shape: {back_array.shape}")
    assert rgb_array.shape == (len(rgbList), 1024, 1024, 3)
    assert mask_array.shape == (len(rgbList), 1024, 1024, 3)
    assert back_array.shape == (len(rgbList), 1024, 1024, 3)

    #invert the mask
    
    #add the background to the image
    #invert the mask
    print (f"invert the mask")
    mask_array = cv.bitwise_not(mask_array)
    #add the background to the image
    print (f"add the background to the image")
    rgb_array[mask_array != 0] = back_array[mask_array != 0]

    #save the images
    for i in tqdm(range(len(rgbList))):
        output_path: str = os.path.join(output_directory, f"mask{rgbList[i][3:]}")

        cv.imwrite(output_path, rgb_array[i])



        


    """
    for imgName in tqdm(rgbList):
        img_path: str = os.path.join(rgb_path, imgName)
        img: np.ndarray = np.array(cv.imread(img_path))
       

        # Read the mask
        #name of the mask is the same as the rgb image withouth the rgb
        imgName = imgName[3:]
        mask_path_2: str = os.path.join(mask_path, imgName)
        mask: np.ndarray = np.array(cv.imread(mask_path_2))
     

        # Get the background
        bg_path: str = os.path.join(backimg_dir_2, f"{starting_index:04d}.png")
      
        bg: np.ndarray = np.array(cv.imread(bg_path))
        starting_index += 1

        # Add the background to the image
        #invert the mask
        mask = cv.bitwise_not(mask)
        img[mask != 0] = bg[mask != 0]

        # Save the image
        output_path: str = os.path.join(output_directory, imgName)
        cv.imwrite(output_path, img)"""


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
