########## Project Description ##########
# This is a python script for deleting the background of the handwritten image.
# Input: Original Image and Handwritten Image
# Output: Handwritten Image without background
# Required Library: OpenCV(install required), time(pre-installed)
# BUILD: Jun 05, 2024 (KST)
##########################################

########## CHANGELOG ##########
# Jun 05, 2024 (KST)
# Delete unnecessary libraries
# Add time library to check the running time
# Revise the code more general
##############################

# Import the necessary library
import cv2
import time
import natsort
import os


class delete_bg:
    def __init__(self, extract_image_path, pjt_name):
        # important!! -> input must be the path of the image file
        self.original_img_path = f'{extract_image_path}/original' # Original Image
        self.handwritten_img_path = f'{extract_image_path}/handwritten' # Handwritten Image
        self.pjt_name = pjt_name # Project Name
        
    def delete_background(self):
        # Report Start time to calculate running time
        start_time = time.time()
        
        # 0. Get the list of the image
        ## Original Image
        original_img_list = os.listdir(self.original_img_path)
        original_img_list = natsort.natsorted(original_img_list)
        ## Handwritten Image
        handwritten_img_list = os.listdir(self.handwritten_img_path)
        handwritten_img_list = natsort.natsorted(handwritten_img_list)
        
        for ori, hand in zip(original_img_list, handwritten_img_list):
            # 1. Read the image
            original_img = cv2.imread(f'{self.original_img_path}/{ori}')
            handwritten_img = cv2.imread(f'{self.handwritten_img_path}/{hand}')
            
            # 2.Convert BGR to RGB(CV2 format to numpy format)
            ## We only need to convert handwritten image
            handwritten = cv2.cvtColor(handwritten_img, cv2.COLOR_BGR2RGB)
            
            # 3. Get Different between Original and Handwritten Image
            diff = cv2.bitwise_not(cv2.absdiff(original_img, handwritten_img))
            diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
            
            # 4. make mask
            mask = (diff_rgb > 225).all(axis=2)
            
            # 5. apply mask
            handwritten[mask] = [255, 255, 255]
            
            # 6. Convert RGB to BGR(numpy format to CV2 format)
            handwritten = cv2.cvtColor(handwritten, cv2.COLOR_RGB2BGR)
            
            # 7. Save the result
            print("Save the result!!")
            ## Make the directory to save the result
            save_DIR = f'Result/{self.pjt_name}/delete_bg'
            if not os.path.exists(save_DIR):
                os.makedirs(save_DIR)
            
            name = f'{hand.split(".")[0]}_delete_bg'
            cv2.imwrite(f'{save_DIR}/{name}.jpg', handwritten)
            print("Complete to save the result!!")
        
        # Report End time to calculate running time
        end_time = time.time()
        print(f"Running Time: {end_time - start_time} seconds")
        # return the value
        return save_DIR