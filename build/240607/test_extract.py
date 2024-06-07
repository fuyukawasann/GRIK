import time
import os
import sys
from Utils.extract import extractor as etr

print("Delete Background Part!!")
time.sleep(1) # wait for 1 seconds
# ### Get the path of the images
# print(f'Current Directory: {os.getcwd()}')
# print(f'Element in the current directory: {os.listdir('Compare')}')
# original_img_path = input("Enter the path of the original image: ")
# handwritten_img_path = input("Enter the path of the handwritten image: ")
### Check the path of the images and if not exit.
# print(f'Current Directory: {os.getcwd()}')
# extract_folder = f'Result/{res_name}/Extracted' # BBOX Extracted module would make this directory
result_name = input("Enter the result name: ")
if not os.path.exists(f'Result/{result_name}/YOLO'):
    print("There's no extracted folder!! Please check the path!!")
    sys.exit()
### Delete the background
etr_obj = etr(f'Result/{result_name}/YOLO', result_name)
etr_result_path, etr_eval_time = etr_obj.extract_handwritten() # Result is the path of the saved image
print("End of the module!!")
time.sleep(1) # wait for 1 seconds