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
    def __init__(self, video_path, res_name):
        # important!! -> input must be the path of the video file
        self.video_path = video_path
        self.res_name = res_name
    
    def SSIMprocessor(self, first, second, thisTurn, iter, output_path):
        ## Filtered Gray Scale
        grayA = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

        ## Saving Score
        (score, diff) = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        ## If score is greater than 0.87, then delete Primary
        if thisTurn == 1:
            cv2.imwrite(f'{output_path}/original/{self.res_name}_{iter}.jpg', first)
            print(f'#{thisTurn - 1} Frame Saved -> {iter}_original')
            return iter
        if score < 0.87:
            cv2.imwrite(f'{output_path}/handwritten/{self.res_name}_{iter}.jpg', first)
            print(f'#{thisTurn - 1} Frame Saved -> {iter}_handwritten')
            cv2.imwrite(f'{output_path}/original/{self.res_name}_{iter+1}.jpg', second)
            print(f'#{thisTurn} Frame Saved -> {iter+1}_original')
            return iter + 1
        else: return iter

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
        output_path = f'Result/{self.res_name}/SSIM' # for testing
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
        iterator = 0
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
                iterator = ssim_cpu.SSIMprocessor(first, second, frame_counter, iterator, output_path)
                first = second
            frame_counter += 1
            ### Save the last frame
            if int(cap.get(1)) == frame_count:
                cv2.imwrite(f'{output_path}/handwritten/{self.res_name}_{iterator}.jpg', frame)
                print(f'#{frame_counter} Frame Saved -> {iterator}_handwritten')
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
        return output_path, eval_time