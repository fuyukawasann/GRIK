########## Project Description ##########
# This is a python script for extracting meaningful frame.
# Input: Path of the video file
# Output: Path of the extracted frame folder
# Required Library: OpenCV(install required), numpy(install required), os(pre-installed)
# BUILD: Jun 05, 2024 (KST)
##########################################

# import the necessary library
import sys
import os
import time
try:
    import cv2
except:
    os.system('pip install opencv-python')
    import cv2


class ssim_gpu:
    def __init__(self, video_path):
		# important!! -> input must be the path of the video file
        self.video_path = video_path
        
    def ssim_gpu_calculation(self):
		# Read the video
        cap = cv2.VideoCapture(self.video_path)
		## Check the video is opened
        if not cap.isOpened():
            print("Error: Video file could not be opened!!")