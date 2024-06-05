
import os
import time

from detection import detection_ps as dps

if __name__ == '__main__':
    print("This is the main part of the file")
    time.sleep(2) # wait for 2 seconds
	
	## Test detection module!!
    print("Test detection module!!")
    time.sleep(1) # wait for 1 seconds
    print(f'Current Directory: {os.getcwd()}')
    # img_path = input("Enter the path of the image: ")
    detection_obj = dps('Images')
    result_path = detection_obj.detection_panseo()
    print(f'Result path: {result_path}')
    print("End of the module!!")
    time.sleep(1) # wait for 1 seconds
	
	## EOF
    print("End of the file!!")