########## Project Description ##########
# This is a python script for our project.
# Required Library: opencv, torch, skimage, numpy, os
# BUILD: Jun 02, 2024 (KST)
##########################################

## import the necessary library
import sys
import os
import cv2
import time

## import my own module
from delete_bg import delete_bg as del_bg





## main part of the file
if __name__ == '__main__':
    print("This is the main part of the file")
    time.sleep(2) # wait for 2 seconds
    
    ## Test delete_bg module!!
    print("Test delete_bg module!!")
    time.sleep(1) # wait for 1 seconds
    print(f'Current Directory: {os.getcwd()}')
    original_img_path = input("Enter the path of the original image: ")
    handwritten_img_path = input("Enter the path of the handwritten image: ")
    del_bg_obj = del_bg(original_img_path, handwritten_img_path)
    result = del_bg_obj.delete_background()
    print("End of the module!!")
    time.sleep(1) # wait for 1 seconds
    
    ## Save the result
    print("Save the result!!")
    time.sleep(2) # wait for 2 seconds
    save_DIR = 'Result'
    if not os.path.exists(save_DIR):
        os.makedirs(save_DIR)
    name = input("Enter the name of the file: ")
    cv2.imwrite(f'{save_DIR}/{name}.jpg', result)
    print("Complete to save the result!!")
    time.sleep(1) # wait for 1 seconds
    
    ## EOF
    print("End of the file!!")
  