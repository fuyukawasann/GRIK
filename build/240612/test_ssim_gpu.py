import os
import shutil
from datetime import datetime
from Utils.ssim_gpu import ssim_gpu as scg

res_name = 'Testerer'

os.system('gdown —id "1lb49qaM—C__1XthD2hO5PpXeLrC5Im5"')
print("GDOWN SUCCESS")

video_path = f'Result/{res_name}/Video'
if not os.path.exists(video_path):
    os.makedirs(video_path)
video_path = f'{video_path}/DEMO.mp4'
shutil.move('DEMO.mp4', video_path)

ssim_obj = scg(video_path, res_name)
save_img_path, ssim_eval_time = ssim_obj.ssim_gpu_calculation()

print(f'Eval Time: {ssim_eval_time}')
print(f'Saved Path: {save_img_path}')
print('Saved Complete')
