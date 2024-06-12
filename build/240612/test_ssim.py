import os
import shutil
from datetime import datetime
from Utils.ssim_cpu import ssim_cpu as scc

res_name = 'Testerer'

# 15WrOYg9Klmt90WYPce5qsXKug4QwgfDC -> DEMO1
# 1lb49qaM--C__1XthD2hO5PpXeLrC5Im5 -> DEMO2

os.system('gdown https://drive.google.com/uc?id=15WrOYg9Klmt90WYPce5qsXKug4QwgfDC')
print("GDOWN SUCCESS")

video_path = f'Result/{res_name}/Video'
if not os.path.exists(video_path):
    os.makedirs(video_path)
video_path = f'{video_path}/DEMO.mp4'
shutil.move('DEMO.mp4', video_path)

ssim_obj = scc(video_path, res_name)
save_img_path, ssim_eval_time = ssim_obj.ssim_cpu_calculation()

print(f'Eval Time: {ssim_eval_time}')
print(f'Saved Path: {save_img_path}')
print('Saved Complete')
