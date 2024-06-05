########## Project Description ##########
# This is a python script that detects the presence of a "panseo" in an image.
# Input: Directory of the saved image
# Output: Directory of the saved image with the detected "panseo"
# Required Library: torch, time, os, cv2, natsort
# BUILD: Jun 04, 2024 (KST)
########################################

########## CHANGE LOG ##########
# Jun 05, 2024 (KST)
# Add checking time system
# Change saving directory
###############################


# Import the necessary library
import torch
import time
import os
import cv2
try:
    import natsort
except ImportError:
    os.system("pip install natsort")
    import natsort


class detection_ps:
	def __init__(self, img_path, result_name):
		self.img_path = img_path
		self.result_name = result_name
        
	def detection_panseo(self):
		print("This is the detection_panseo module")
		time.sleep(2)
	
		# Check if CUDA is available
		if torch.cuda.is_available():
			print("CUDA is available!! Running on CUDA...")
			device_type = "cuda"
		## If CUDA is not available, check if this device is mac
		elif(torch.backends.mps.is_available()):
			print("MPS is available!! Running on MPS...")
			device_type = "mps"
		## If CUDA is not available and not a mac, run on CPU
		else:
			print("CUDA is not available!! Running on CPU...")
			device_type = "cpu"

		# Load the model
		print("Load Model")
		time.sleep(2)

		# Check the path of the YOLOv7 model
		# print("Warning: IF you don't pre-download the model, please enter 'NONE' in the path")
		print("Check the path of the YOLOv7 model and if not exist, download the model")
		time.sleep(1)

		current_folder_list = os.listdir(os.getcwd())
		if('yolov7' not in current_folder_list):
			print("There's no model -> Download the model!!")
			time.sleep(1)
			os.system("git clone https://github.com/WongKinYiu/yolov7.git")
		else:
			print("There's a model in the current folder!!")
			time.sleep(1)
		

		# path_yolov7 = input("Enter the path of the YOLOv7 model: ")
  
		# if(path_yolov7 == "NONE"):
		# 	print("There's no model -> Download the model!!")
		# 	time.sleep(1)
		# 	os.system("git clone https://github.com/WongKinYiu/yolov7.git")
		# 	path_yolov7 = "./yolov7"

		# Select weight files
		print("Select Weight File")
		time.sleep(1)
		## Check the wrong selection of the weight file
		while True:
			print(f'Exist Weight Files: {os.listdir("Weights")}')
			name_weight = input("Enter the name of the weight file(include Extensions): ")
			if(name_weight in os.listdir("Weights")):
				break
			else:
				print("Wrong selection of the weight file!!")
				print("Please select the weight file again!!")
				time.sleep(1)
		## If the weight file is correct, load the model
		pretrained_model = torch.hub.load('yolov7', 'custom', f'Weights/{name_weight}', source='local')
		pretrained_model = pretrained_model.to(device_type)

		# Image Setting and save setting
		list_img = os.listdir(self.img_path)
		list_img = natsort.natsorted(list_img)
		print(f'List of images: {list_img}')
		save_img_path = f'Result/{self.result_name}/Result_Panseo' # Before, 'Result_Panseo'
		if not os.path.exists(save_img_path):
			os.makedirs(save_img_path)

		# Processing the image
		print("Processing the image")
		time.sleep(2)
		## Check the start time
		strat_time = time.time()
		for img in list_img:
			img_name = img.split('.')[0]
			this_img = cv2.imread(f'{self.img_path}/{img}')
			result_temp = pretrained_model(f'{self.img_path}/{img}')
			xyxys = result_temp.pandas().xyxy[0].values # When test delete '.values'
			# print(xyxys)
			# xyxys = xyxys.values
			for this_xyxys in xyxys:
				x, y, x2, y2, confi, cls_num, cls_name = this_xyxys
				# print(x, type(x))
				cv2.rectangle(this_img, (int(x), int(y)), (int(x2), int(y2)), (0,0,255), 2)
				cv2.putText(this_img, cls_name, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
				cv2.putText(this_img, str(round(confi, 2)), (int(x), int(y)-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    
			cv2.imwrite(f'{save_img_path}/{img_name}_detect.jpg', this_img)

		## Check the end time
		end_time = time.time()
		print(f"Running Time: {end_time - strat_time} seconds")
		time.sleep(1)
		print("End of the detection_panseo module")
		time.sleep(2)
		return save_img_path


