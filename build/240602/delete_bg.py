########## Project Description ##########
# This is a python script for deleting the background of the handwritten image.
# Input: Original Image and Handwritten Image
# Output: Handwritten Image without background
# Required Library: OpenCV(install required), numpy(install required), os(pre-installed)
# BUILD: Jun 02, 2024 (KST)
##########################################

# Import the necessary library
import cv2
import numpy as np
import os


class delete_bg:
    def __init__(self, original_img, handwritten_img):
        # important!! -> input must be the path of the image file
        self.original_img = original_img
        self.handwritten_img = handwritten_img
        
    def delete_background(self):
        # Read the image
        self.original_img = cv2.imread(self.original_img)
        self.handwritten_img = cv2.imread(self.handwritten_img)
        
        # Convert BGR to RGB(CV2 format to numpy format)
        ## We only need to convert handwritten image
        handwritten = cv2.cvtColor(self.handwritten_img, cv2.COLOR_BGR2RGB)
        
        # Get Different between Original and Handwritten Image
        diff = cv2.bitwise_not(cv2.absdiff(self.original_img, self.handwritten_img))
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        
        # make mask
        mask = (diff_rgb > 225).all(axis=2)
        
        # apply mask
        handwritten[mask] = [255, 255, 255]
        
        # Convert RGB to BGR(numpy format to CV2 format)
        handwritten = cv2.cvtColor(handwritten, cv2.COLOR_RGB2BGR)
        
        # return the value
        return handwritten