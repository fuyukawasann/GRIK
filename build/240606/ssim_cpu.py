########## Project Description ##########
# This is a python script for extracting meaningful frame.
# Input: Path of the video file
# Output: Path of the extracted frame folder
# Required Library: OpenCV(install required), numpy(install required), os(pre-installed)
# BUILD: Jun 04, 2024 (KST)
##########################################

# import the necessary library
import sys
import os
from PIL import Image
from datetime import datetime
import shutil
import time
try:
    import cv2
except:
    os.system('pip install opencv-python')
    import cv2
try:
    from skimage.metrics import structural_similarity as ssim
except:
    os.system('pip install scikit-image')
    from skimage.metrics import structural_similarity as ssim


class ssim_cpu:
    def __init__(self, video_path):
        # important!! -> input must be the path of the video file
        self.video_path = video_path
    
    def SSIMprocessor(first, second, thisTurn, output_path):
        ## Filtered Gray Scale
        grayA = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

        ## Saving Score
        (score, diff) = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        ## If score is greater than 0.87, then delete Primary
        if thisTurn == 1:
            cv2.imwrite(f'{output_path}/frame{thisTurn - 1}.jpg', first)
            print(f'#{thisTurn - 1} Frame Saved!!')
        if score < 0.87:
            cv2.imwrite(f'{output_path}/frame{thisTurn - 1}.jpg', first)
            print(f'#{thisTurn - 1} Frame Saved!!')
            cv2.imwrite(f'{output_path}/diff{thisTurn}.jpg', second)
            print(f'#{thisTurn} Frame Saved!!')

    def ssim_cpu_calculation(self):
        # Read the video
        cap = cv2.VideoCapture(self.video_path)
        ## Check the video is opened
        if not cap.isOpened():
            print("Error: Video file could not be opened!!") # Failed
            sys.exit()
        print("Video file is opened successfully!!") # Success
        time.sleep(1)

        # Get Basic Video Information
        ## Get the frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ## Get the frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ## Get the frame rate
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        ## Get the codec
        codec = cv2.VideoWriter_fourcc(*'XVID')
        print("Getting the video information is completed!!")
        time.sleep(1)

        # Get the output path
        ## Define the output path
        output_path = 'Extracted_Frames' # for testing
        # output_path = 'Images'
        ## Check the output path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print("Output path is created successfully!!")
        else: print("Output path is already exist!!")
        time.sleep(1)
        print("Output path is defined successfully!!")
        time.sleep(1)
        
        # SSIM
        ## Check Start Time
        start_time = time.time()
        ## Define Frame Counter
        frame_counter = 0
        ## Video Processing
        while(cap.isOpened()):
            ret, frame = cap.read()
            ### Check the frame is read
            if not ret:
                break
            ### Check the frame is the first frame
            if frame_counter == 0:
                first = frame
            else:
                second = frame
                ssim_cpu.SSIMprocessor(first, second, frame_counter, output_path)
                first = second
            frame_counter += 1
            ### Save the last frame
            if int(cap.get(1)) == frame_count:
                cv2.imwrite(f'{output_path}/frame{frame_counter}.jpg', frame)
                break
        
        # Release the video
        cap.release()
        ## Check End Time
        end_time = time.time()
        print(f"Running Time: {end_time - start_time} seconds")
        time.sleep(1)
        # Return the output path
        print("SSIM Calculation is completed!!")
        return output_path