########## Project Description ##########
# This is a python script for our project.
# Required Library: opencv, torch, skimage, numpy, os
# BUILD: Jun 02, 2024 (KST)
##########################################

########## Change Log ##########
# Jun 04, 2024 (KST)
# Add Object Detection Part!!
###############################

## import the necessary library
import torch
import os
import cv2
import time
import platform

## import my own module
from ssim_cpu import ssim_cpu as scc
from delete_bg import delete_bg as del_bg
from detection import detection_ps as dps




## main part of the file
if __name__ == '__main__':
    ## Intro_message
    print("This is the main part of the file")
    time.sleep(2) # wait for 2 seconds
    
    ## 1. SSIM - Mark the best frame
    ### Inform the user
    print("SSIM Part!!")
    time.sleep(1) # wait for 1 seconds
    ### USE Different module depending on the platform
    if(torch.cuda.is_available()):
        # Use GPU version of the SSIM module
        # print("CUDA is available!!")
        # time.sleep(1) # wait for 1 seconds
        # print("Use the CUDA version of the SSIM module!!")
        # time.sleep(1)
        # #### Input the path of the video file
        # print("Current Directory: ", os.getcwd())
        # video_path = input("Enter the path of the video file: ")
        # ssim_obj = scc(video_path)
        # save_img_path = ssim_obj.ssim_cpu_calculation()
        # print(f'Saved Image Path: {save_img_path}')
        print('Saved Complete')
    else:
        print("CUDA is not available!!")
        time.sleep(1) # wait for 1 seconds
        print("Use the CPU version of the SSIM module!!")
        time.sleep(1)
        #### Input the path of the video file
        print("Current Directory: ", os.getcwd())
        video_path = input("Enter the path of the video file: ")
        ssim_obj = scc(video_path)
        save_img_path = ssim_obj.ssim_cpu_calculation()
        print(f'Saved Image Path: {save_img_path}')
        print('Saved Complete')

    ## 2. Object Detection - Detect the handwritten part
    ### Inform the user
    print("Object Detection Part!!")
    time.sleep(1) # wait for 1 seconds
    ### Object detection module!!
    print("Test detection module!!")
    time.sleep(1) # wait for 1 seconds
    print(f'Current Directory: {os.getcwd()}')
    detection_obj = dps('Images')
    result_path = detection_obj.detection_panseo()
    print(f'Result path: {result_path}')
    print("End of the module!!")
    time.sleep(1) # wait for 1 seconds

    ## 3. Delete Background - Delete the background of the handwritten part
    ### Inform the user
    print("Delete Background Part!!")
    time.sleep(1) # wait for 1 seconds
    ### Get the path of the images
    print(f'Current Directory: {os.getcwd()}')
    print(f'Element in the current directory: {os.listdir('Compare')}')
    original_img_path = input("Enter the path of the original image: ")
    handwritten_img_path = input("Enter the path of the handwritten image: ")
    ### Delete the background
    del_bg_obj = del_bg(original_img_path, handwritten_img_path)
    result = del_bg_obj.delete_background()
    print("End of the module!!")
    time.sleep(1) # wait for 1 seconds
    ### Save the result
    print("Save the result!!")
    time.sleep(2) # wait for 2 seconds
    save_DIR = 'Result'
    if not os.path.exists(save_DIR):
        os.makedirs(save_DIR)
    name = input("Enter the name of the file: ")
    cv2.imwrite(f'{save_DIR}/{name}.jpg', result)
    print("Complete to save the result!!")
    time.sleep(1) # wait for 1 seconds
    
    ## 4. Delete YOLOv7
    ### Inform the user
    print("Delete YOLOv7 repository!!")
    time.sleep(1) # wait for 1 seconds
    ### Delete the repository
    my_plat = platform.system()
    if my_plat == 'Windows':
        os.system('rmdir /s yolov7')
    else:
        os.system('rm -rf yolov7')
    ## EOF
    print("End of the file!!")
  