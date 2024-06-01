## This python file is used to test the function of extract_handwrite_ver1.ipynb

# Import the necessary library
import cv2
import numpy as np
import os

## Convert Image to CV2 format
def cv2_image(image_path):
    img = cv2.imread(image_path)
    return img

def diff_and_rgb_image(original_image, handwrite_image):
    diff = cv2.bitwise_not(cv2.absdiff(original_image, handwrite_image))
    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
    return diff_rgb

def process_img(filter, diff_rgb):
    # Make white img
    new_img = np.zeros_like(diff_rgb)

    # Set Condition
    condition = (diff_rgb > 225).all(axis=2)

    # Apply Condition
    new_img[condition] = diff_rgb[condition]
    new_img[~condition] = filter

    # Convert to CV2 format
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    return new_img

## Main function
if __name__ == '__main__':
    print("Extract Handwrite Image Start")
    # Path of images
    original_path = input('Enter the path of the original image: ')
    handwrite_path = input('Enter the path of the handwrite image: ')

    # Preprocess the image
    original_image = cv2_image(original_path)
    handwrite_image = cv2_image(handwrite_path)

    # Get Handwrite Part
    diff_rgb = diff_and_rgb_image(original_image, handwrite_image)

    # Initialize Filter
    new_color = (255, 0, 0) # Red - Numpy

    # Apply Filter and Convert to CV2 format
    new_img = process_img(new_color, diff_rgb)

    # Check save directory and if not exist, create it
    save_DIR = 'Result'
    try:
        if not os.path.exists(save_DIR):
            os.makedirs(save_DIR)
    except OSError:
        print(f'Error: Cannot Find SAVE_PATH: {save_DIR}')
    name = input('Enter the name of the result image: ')
    # Save the image
    cv2.imwrite(f'{save_DIR}/{name}.jpg', new_img)
    print("Extract Handwrite Image Done")
