########## Project Description ##########
# This is a python script for extracting meaningful frame.
# Input: Path of the video file
# Output: Path of the extracted frame folder
# Required Library: OpenCV(install required), numpy(install required), os(pre-installed)
# BUILD: Jun 04, 2024 (KST)
##########################################

########## Change Log ##########
# Jun 06, 2024 (KST)
# Add the code that comparing running time of each sections
###############################

# import the necessary library
import sys
import os
from PIL import Image
from datetime import datetime
import shutil
import time
import cv2
try:
    from skimage.metrics import structural_similarity as ssim
except:
    os.system('pip install scikit-image')
    from skimage.metrics import structural_similarity as ssim


class ssim_cpu:
    def __init__(self, video_path, res_name):
        # important!! -> input must be the path of the video file
        self.video_path = video_path
        self.res_name = res_name
    
    def SSIMprocessor(self, first, second, thisTurn, iterator):
        ## Filtered Gray Scale
        # print(f'Processing Frame #{thisTurn}...')
        # print(f'Output Path: {self.output_path}')
        grayA = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

        ## Saving Score
        (score, diff) = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        ## If score is greater than 0.87, then delete Primary
        if thisTurn == self.frame_rate * 2:
            cv2.imwrite(f'{self.output_path}/original/{self.res_name}_{iterator}.jpg', first)
            print(f'#{thisTurn - self.frame_rate} Frame Saved -> {iterator}_original')
            return iterator
        if score < 0.929: # before 0.8 and 0.87
            cv2.imwrite(f'{self.output_path}/handwritten/{self.res_name}_{iterator}.jpg', first)
            print(f'#{thisTurn - self.frame_rate} Frame Saved -> {iterator}_handwritten')
            cv2.imwrite(f'{self.output_path}/original/{self.res_name}_{iterator+1}.jpg', second)
            print(f'#{thisTurn} Frame Saved -> {iterator+1}_original')
            return iterator + 1
        else: return iterator

    def ssim_cpu_calculation(self):
        # Read the video
        print("Reading the video...")
        time.sleep(1)
        cap = cv2.VideoCapture(self.video_path)
        ## Check the video is opened
        if not cap.isOpened():
            print("Error: Video file could not be opened") # Failed
            sys.exit()
        print("Reading the video... SUCCESS") # Success
        time.sleep(1)

        # Get Basic Video Information
        print("Getting the video information...")
        ## Get the frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ## Get the frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ## Get the frame rate
        self.frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
        print(f"Frame Count: {frame_count}")
        time.sleep(1)
        ## Get the codec
        codec = cv2.VideoWriter_fourcc(*'XVID')
        print("Getting the video information... SUCCESS")
        time.sleep(1)

        # Get the output path
        print("Defining the output path...")
        ## Define the output path
        self.output_path = f'Result/{self.res_name}/SSIM' # for testing
        # output_path = 'Images'
        ## Check the output path
        if not os.path.exists(self.output_path):
            print("INFO: Output path is not exist")
            print("Creating the output path...")
            time.sleep(1)
            os.makedirs(self.output_path)
            os.makedirs(f'{self.output_path}/original')
            os.makedirs(f'{self.output_path}/handwritten')
            print("Creating the output path... SUCCESS")
            time.sleep(1)
        else: 
            print("INFO: Output path is already exist")
            time.sleep(1)
        print("Defining the output path... SUCCESS")
        time.sleep(1)
        
        # SSIM
        ## Check Start Time
        print("SSIM Calculation...")
        start_time = time.time()
        ## Define Frame Counter
        # frame_counter = 0
        iterator = 0
        ## Video Processing
        while(cap.isOpened()):
            ret, frame = cap.read()
            ### Check the frame is read
            # if not ret:
            #     print("Error: Frame could not be read")
            #     break
            ### Check the frame is the first frame
            if (int(cap.get(1)) % self.frame_rate == 0):
                if iterator == 0:
                    first = frame
                    iterator += 1
                else:
                    second = frame
                    iterator = self.SSIMprocessor(first, second, cap.get(1), iterator)
                    first = second
                # frame_counter += 1
                ### Save the last frame
            if int(cap.get(1)) == frame_count:
                cv2.imwrite(f'{self.output_path}/handwritten/{self.res_name}_{iterator}.jpg', frame)
                print(f'#{cap.get(1)} Frame Saved -> {iterator}_handwritten')
                break
        
        # Release the video
        cap.release()
        ## Check End Time
        end_time = time.time()
        eval_time = end_time - start_time
        print(f"Running Time: {eval_time} seconds")
        time.sleep(1)
        # Return the output path
        print("SSIM Calculation is completed!!")
        return self.output_path, eval_time
