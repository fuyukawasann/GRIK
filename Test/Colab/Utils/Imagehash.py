########## Project Description ##########
# This is a python script for extracting meaningful frame.
# Input: Path of the video file
# Output: Path of the extracted frame folder
# Required Library: OpenCV(install required), numpy(install required), os(pre-installed)
# BUILD: Jun 04, 2024 (KST)
##########################################

########## Change Log ##########
# Jun 13, 2024 (KST)
# Using OpenCV to acclelerate the SSIM calculation
###############################

# import the necessary library
import sys
import os
from PIL import Image
import time
import cv2
import imagehash

class Imagehash:
    def __init__(self, video_path, res_name):
        # important!! -> input must be the path of the video file
        self.video_path = video_path
        self.res_name = res_name
        cv2.setUseOptimized(True)
        cv2.setNumThreads(0)

    def imagehash_calculation(self):
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
        ## Get the frame rate
        self.frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
        print(f"Frame Count: {frame_count}")
        time.sleep(1)
        ## Get the codec
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
        
        # Imagehash
        ## Check Start Time
        print("Imagehash Calculation...")
        start_time = time.time()
        ## Define Frame Counter
        # frame_counter = 0
        iterator = 0
        ## Video Processing
        while(cap.isOpened()):
            ret, frame = cap.read()
            if (int(cap.get(1)) % self.frame_rate == 0):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if iterator == 0:
                    first = frame
                    first_PIL = Image.fromarray(first)
                    first_hash = imagehash.phash(first_PIL)
                    iterator += 1
                    first_PIL.save(f'{self.output_path}/original/{self.res_name}_{iterator}.jpg')
                    print(f'#{cap.get(1)} Frame Saved -> {iterator}_original')
                else:
                    second = frame
                    second_PIL = Image.fromarray(second)
                    second_hash = imagehash.phash(second_PIL)
                    inner_start_time = time.time() # Check the time
                    distance = first_hash - second_hash
                    inner_end_time = time.time() # Check the time
                    print(f"Imagehash Calculation Time: {inner_end_time - inner_start_time} seconds") # Check the time
                    if(distance >= 10):
                        iterator += 1
                        first_PIL.save(f'{self.output_path}/handwritten/{self.res_name}_{iterator - 1}.jpg')
                        print(f'#{cap.get(1)-1} Frame Saved -> {iterator - 1}_handwritten')
                        second_PIL.save(f'{self.output_path}/original/{self.res_name}_{iterator}.jpg')
                        print(f'#{cap.get(1)} Frame Saved -> {iterator}_original')
                    first_PIL = second_PIL
                    first_hash = second_hash
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
        print("Imagehash Calculation is completed!!")
        return self.output_path, eval_time
