
import os
import time

from detection import detection_ps as dps

if __name__ == '__main__':
    print("This is the main part of the file")
    time.sleep(2) # wait for 2 seconds
	
	## Test detection module!!
    print("Test detection module!!")
    time.sleep(1) # wait for 1 seconds
    res_name = input("Enter the name of the result: ")
    print(f'Current Directory: {os.getcwd()}')
    # img_path = input("Enter the path of the image: ")
    detection_obj = dps('Images', res_name)
    result_path, db_eval_time = detection_obj.detection_panseo()
    print(f'Result path: {result_path}')
    print(f'Evaluation time: {db_eval_time}')
    print("End of the module!!")
    time.sleep(1) # wait for 1 seconds
	
	## EOF
    print("End of the file!!")