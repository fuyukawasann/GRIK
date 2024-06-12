########## Project Description ##########
# This is a python script for our project.
# Required Library: torch, os, datetime, time, platform, sys and my own modules
# BUILD: Jun 02, 2024 (KST)
##########################################

########## Change Log ##########
# Jun 12, 2024 (KST)
# Delete ssim_gpu.py because we failed to build the module
###############################

## import the necessary library
import torch
import os
from datetime import datetime
import time
import platform
import sys
import shutil

## import my own module
from Utils.ssim_cpu import ssim_cpu as scc
from Utils.extract import extractor as etr
from Utils.detection_trt import detection_ps_trt as dtrt
#from Utils.detection import detection_ps as dps
from Utils.makePDF import makePDF as mpdf




## main part of the file
if __name__ == '__main__':
    ## Intro_message
    print("This is the main part of the file")
    time.sleep(2) # wait for 2 seconds
    
    ## 0. Get Saved Name
    print("Get the name of the result...")
    time.sleep(1)
    res_name = input("Enter the name of the result: ")
    ### Add the current time to the name
    res_name = f'{res_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    ### Inform the user about the name
    print(f'The name of the result(Include datetime!!): {res_name}')
    print("Get the name of the result... SUCCESS")
    time.sleep(1) # wait for 1 seconds
    

    ## 0-1. Download Video.mp4
    print("Download video and move to correct directory...")
    time.sleep(1)
    ### Download at GRIK/build/200000
    os.system('gdown —id "1lb49qaM—C__1XthD2hO5PpXeLrC5Im5"')
    ### Make Directory
    video_path = f'Result/{res_name}/Video'
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    video_path = f'{video_path}/DEMO.mp4'
    ### Move video
    shutil.move('DEMO.mp4', video_path)
    print("Donwload video and move to correct directory... SUCCESS")
    time.sleep(1)

    
    ## 1. SSIM - Mark the best frame
    ### Inform the user
    print("SSIM Part...")
    time.sleep(1)
    #### Input the path of the video file
    print("Current Directory: ", os.getcwd())
    #video_path = input("Enter the path of the video file: ")
    ssim_obj = scc(video_path, res_name)
    save_img_path, ssim_eval_time = ssim_obj.ssim_cpu_calculation()
    print(f'Saved Image Path: {save_img_path}')
    print('SSIM Part... SUCCESS')
    time.sleep(1)

    ## 2. Object Detection - Detect the handwritten part
    ### Inform the user
    print("Object Detection Part...")
    time.sleep(1) # wait for 1 seconds
    ### Object detection module!!
    print("Test detection module!!")
    time.sleep(1) # wait for 1 seconds
    print(f'Current Directory: {os.getcwd()}')
    detection_obj = dtrt(save_img_path, res_name) # After, you need to change 'Images' -> Real Image directory
    ob_result_path, ob_eval_time = detection_obj.detection_panseo_trt()
    print(f'Result path: {ob_result_path}')
    print("Object Detection Part... SUCCESS")
    time.sleep(1) # wait for 1 seconds

    ## 3. Delete Background - Delete the background of the handwritten part
    ### Inform the user
    print("Delete Background Part...")
    time.sleep(1) # wait for 1 seconds
    # ### Get the path of the images
    # print(f'Current Directory: {os.getcwd()}')
    # print(f'Element in the current directory: {os.listdir('Compare')}')
    # original_img_path = input("Enter the path of the original image: ")
    # handwritten_img_path = input("Enter the path of the handwritten image: ")
    ### Check the path of the images and if not exit.
    # print(f'Current Directory: {os.getcwd()}')
    # extract_folder = f'Result/{res_name}/Extracted' # BBOX Extracted module would make this directory
    if not os.path.exists(ob_result_path):
        print("There's no extracted folder!! Please check the path!!")
        sys.exit()
    ### Delete the background
    etr_obj = etr(ob_result_path, res_name)
    etr_result_path, etr_eval_time = etr_obj.extract_handwritten() # Result is the path of the saved image
    print("Delete Background Part... SUCCESS")
    time.sleep(1) # wait for 1 seconds
    
    ## 4. Make PDF File
    ### Inform the user
    print("Make PDF File Part...")
    time.sleep(1)
    ### Make PDF
    mpdf_obj = mpdf(save_img_path, etr_result_path, res_name)
    mpdf_eval_time = mpdf_obj.make_pdf()
    print("Make PDF File Part... SUCCESS")
    
    
    ## 5. Print Out the Running Time
    print("Print Out the Running Time...")
    print("=====================================")
    time.sleep(1) # wait for 1 seconds
    ### SSIM
    print(f'SSIM Running Time: {ssim_eval_time} seconds')
    ### Object Detection
    print(f'Object Detection Running Time: {ob_eval_time} seconds')
    ### Delete Background
    print(f'Delete Background Running Time: {etr_eval_time} seconds')
    ### Make PDF File
    print(f'Make PDF File Running Time: {mpdf_eval_time} seconds')
    ### Print Out the Total Running Time
    total_time = ssim_eval_time + ob_eval_time + etr_eval_time + mpdf_eval_time
    print(f'Total Running Time: {total_time} seconds')
    print("=====================================")
    print("Print Out the Running Time... SUCCESS")
    time.sleep(1) # wait for 1 seconds
    
    ## 6. Delete YOLOv7
    ### Inform the user
    print("Delete YOLOv7 repository...")
    time.sleep(1) # wait for 1 seconds
    ### Delete the repository
    my_plat = platform.system()
    if my_plat == 'Windows':
        os.system('rmdir /s yolov7')
    else:
        os.system('rm -rf yolov7')
    ### Finish
    print("Delete YOLOv7 repository... SUCCESS")
    time.sleep(1) # wait for 1 seconds
    
    ## EOF
    print("End of the file!!")
  
