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
# try:
#     from skimage.metrics import structural_similarity as ssim
# except:
#     os.system('pip install scikit-image')
#     from skimage.metrics import structural_similarity as ssim
from __future__ import annotations
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
try:
    import pycuda.gpuarray as gpuarray
except SyntaxError as e:
    if "future feature annotations is not defined" in str(e):
        # 타입 힌팅 기능이 활성화되어 있는 경우 비활성화
        import pycuda.gpuarray as gpuarray
    else:
        raise e
from pycuda.compiler import SourceModule

## define cuda kernel
mod = SourceModule("""
#include <math.h>

__global__ void ssim_kernel(float *img1, float *img2, float *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int index = y * width + x;
        
        float mu1 = 0, mu2 = 0, sigma1 = 0, sigma2 = 0, sigma12 = 0;
        float C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
        
        // 평균, 분산, 공분산 계산
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int x_n = min(max(x + j, 0), width - 1);
                int y_n = min(max(y + i, 0), height - 1);
                int index_n = y_n * width + x_n;
                
                mu1 += img1[index_n];
                mu2 += img2[index_n];
                sigma1 += img1[index_n] * img1[index_n];
                sigma2 += img2[index_n] * img2[index_n];
                sigma12 += img1[index_n] * img2[index_n];
            }
        }
        
        mu1 /= 9;
        mu2 /= 9;
        sigma1 = sqrt(sigma1 / 9 - mu1 * mu1);
        sigma2 = sqrt(sigma2 / 9 - mu2 * mu2);
        sigma12 = (sigma12 / 9) - mu1 * mu2;
        
        // SSIM 계산
        out[index] = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 * sigma1 + sigma2 * sigma2 + C2));
    }
}
""")



class ssim_gpu:
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
        score = self.compute_ssim(grayA, grayB)
        # (score, diff) = ssim(grayA, grayB, full=True)
        #diff = (diff * 255).astype("uint8")

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
        
    def compute_ssim(self, img1, img2):
        # GPU 메모리에 이미지 데이터 복사
        img1_gpu = gpuarray.to_gpu(img1.astype(np.float32))
        img2_gpu = gpuarray.to_gpu(img2.astype(np.float32))
        
        # 출력 배열 생성
        out_gpu = gpuarray.empty_like(img1_gpu)
        
        # CUDA 커널 실행
        block_size = (16, 16)
        grid_size = (
            (img1.shape[1] + block_size[0] - 1) // block_size[0],
            (img1.shape[0] + block_size[1] - 1) // block_size[1]
        )
        ssim_kernel = mod.get_function("ssim_kernel")
        ssim_kernel(img1_gpu, img2_gpu, out_gpu, np.int32(img1.shape[1]), np.int32(img1.shape[0]), block=block_size, grid=grid_size)
        
        # GPU 메모리에서 SSIM 값 가져오기
        ssim = out_gpu.get()
        return ssim

    def ssim_gpu_calculation(self):
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
