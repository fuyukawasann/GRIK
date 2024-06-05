########## Project Description ##########
# This is a python script for our project.
# Required Library: torch, os, datetime, time, platform, sys and my own modules
# BUILD: Jun 02, 2024 (KST)
##########################################

########## Change Log ##########
# Jun 05, 2024 (KST)
# Revise delete_bg part to make it more general
# Add name to save PDF File
###############################

## import the necessary library
import torch
import os
from datetime import datetime
import time
import platform
import sys

## import my own module
from ssim_cpu import ssim_cpu as scc
from ssim_gpu import ssim_gpu as scg
from delete_bg import delete_bg as del_bg
from detection import detection_ps as dps




## main part of the file
if __name__ == '__main__':
    ## Intro_message
    print("This is the main part of the file")
    time.sleep(2) # wait for 2 seconds
    
    ## 0. Get Saved Name
    res_name = input("Enter the name of the result: ")
    ### Add the current time to the name
    res_name = f'{res_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    ### Inform the user about the name
    print(f'The name of the result(Include datetime!!): {res_name}')
    time.sleep(1) # wait for 1 seconds
    
    ## 1. SSIM - Mark the best frame
    ### Inform the user
    print("SSIM Part!!")
    time.sleep(1) # wait for 1 seconds
    ### USE Different module depending on the platform
    if(torch.cuda.is_available()):
        # Use GPU version of the SSIM module
        print("CUDA is available!!")
        time.sleep(1) # wait for 1 seconds
        print("Use the CUDA version of the SSIM module!!")
        time.sleep(1)
        #### Input the path of the video file
        print("Current Directory: ", os.getcwd())
        video_path = input("Enter the path of the video file: ")
        ssim_obj = scg(video_path)
        save_img_path = ssim_obj.ssim_gpu_calculation()
        print(f'Saved Image Path: {save_img_path}')
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
    detection_obj = dps('Images', res_name) # After, you need to change 'Images' -> Real Image directory
    result_path = detection_obj.detection_panseo()
    print(f'Result path: {result_path}')
    print("End of the module!!")
    time.sleep(1) # wait for 1 seconds

    ## 3. Delete Background - Delete the background of the handwritten part
    ### Inform the user
    print("Delete Background Part!!")
    time.sleep(1) # wait for 1 seconds
    # ### Get the path of the images
    # print(f'Current Directory: {os.getcwd()}')
    # print(f'Element in the current directory: {os.listdir('Compare')}')
    # original_img_path = input("Enter the path of the original image: ")
    # handwritten_img_path = input("Enter the path of the handwritten image: ")
    ### Check the path of the images and if not exit.
    print(f'Current Directory: {os.getcwd()}')
    extract_folder = f'Result/{res_name}/Extracted' # BBOX Extracted module would make this directory
    if not os.path.exists(extract_folder):
        print("There's no extracted folder!! Please check the path!!")
        sys.exit()
    ### Delete the background
    del_bg_obj = del_bg(extract_folder, res_name)
    result = del_bg_obj.delete_background() # Result is the path of the saved image
    print("End of the module!!")
    time.sleep(1) # wait for 1 seconds
    ### Save the result
    
    ## 4. Make PDF File
    ###
    
    ## 5. Delete YOLOv7
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
  