########## Project Description ##########
# This is a test script for YOLOv7-tiny model.
# Required files: GRIKDataset.zip
# BUILD: May 31, 2024 (KST)
##########################################

## 0. Import Libraries
import os, sys
import torch
import platform
import shutil
import subprocess
try:
	import wget
except:
    # Upgrade pip module
    subprocess.check_call([sys.executable,'-m', 'pip', 'install', '--upgrade', 'pip'])
    # Install wget module
    subprocess.check_call([sys.executable,'-m', 'pip', 'install', '--upgrade', 'wget'])
    import wget

## 1. Download YOLOv7 Official Repository
def download_yolov7():
    # Download the model
    print("Downloading the repo...")
    os.system("git clone https://github.com/WongKinYiu/yolov7.git")
    print("Repo is downloaded successfully.")

    # Change the directory
    print("Changing the directory...")
    os.chdir("yolov7")
    print("Directory is changed successfully.")


## 2. Install Required Libraries
def install_requirements():
    # Install required libraries
    print("Installing required libraries...")
    os.system("pip install -r requirements.txt")
    print("All libraries are installed successfully.")

## 3. Check current Directory
def check_directory():
    # Check the current directory
    print("Current Directory: ", os.getcwd())

## 4. Check Current System
def check_system():
    # Check the current system
    print("Current System: ", sys.platform)
    print("Pytorch Version: ", torch.__version__)
    print("Python Version: ", sys.version)
    print("Check system is all done.")

## 5. Check the Platform
def check_platform():
    # Check the platform
    my_platform = platform.system()
    print("Platform: ", my_platform)
    return my_platform

## 6. Unzip dataset
def unzip_dataset(platform):
    # Unzip the dataset
    print("Unzipping the dataset...")
    if platform == "Windows":
        print("Windows OS is detected.")
        file_name = "../GRIKDataset.zip"
        output_dir = "data/"
        format = "zip"
        shutil.unpack_archive(file_name, output_dir, format)
    else:
        print("Linux OS or macOS is detected.")
        os.system("unzip -uq '../GRIKDataset.zip' -d 'data/'")
    print("Dataset is unzipped successfully.")

## 7. Download weight file
def download_weight_file(sel_model):
    if(sel_model == '1'):
        print("Downloading the weight file for yolov7...")
        url = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt'
    else:
        print("Downloading the weight file for yolov7-tiny...")
        url = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt'
    wget.download(url)
    print("Weight file is downloaded successfully.")

## 8-1. Training Model(macOS)
def train_model_mac(sel_model):
    while 1:
        mysel = input("Do you want to freeze? 1: Yes 2: No >>")
        if(mysel == '1' or mysel == '2'):
            break
    ## Freeze
    if(mysel == '1'):
        if(sel_model == '1'):
            print("Training the yolov7 model...")
            if(torch.backends.mps.is_available()):
                os.system("python train.py --workers 1 --device mps --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt --freeze 50")
            else:
                os.system("python train.py --workers 1 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt --freeze 50")
        else:
            print("Training the yolov7-tiny model...")
            if(torch.backends.mps.is_available()):
                os.system("python train.py --workers 1 --device mps --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-tiny-custom --weights yolov7-tiny.pt --freeze 50")
            else:
                os.system("python train.py --workers 1 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-tiny-custom --weights yolov7-tiny.pt --freeze 50")
    ## Not Freeze
    else:
        if(sel_model == '1'):
            print("Training the yolov7 model...")
            if(torch.backends.mps.is_available()):
                os.system("python train.py --workers 1 --device mps --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt")
            else:
                os.system("python train.py --workers 1 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt")
        else:
            print("Training the yolov7-tiny model...")
            if(torch.backends.mps.is_available()):
                os.system("python train.py --workers 1 --device mps --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-tiny-custom --weights yolov7-tiny.pt")
            else:
                os.system("python train.py --workers 1 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-tiny-custom --weights yolov7-tiny.pt")
    print("Training is done.")

## 8-2. Training Model(Windows or Linux)
def train_model(sel_model):
    while 1:
        mysel = input("Do you want to freeze? 1: Yes 2: No >>")
        if(mysel == '1' or mysel == '2'):
            break
    if(mysel == '1'):
        if(sel_model == '1'):
            print("Training the yolov7 model...")
            if(torch.cuda.is_available()):
                os.system("python train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt --freeze 50")
            else:
                os.system("python train.py --workers 1 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt --freeze 50")
        else:
            print("Training the yolov7-tiny model...")
            if(torch.cuda.is_available()):
                os.system("python train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-tiny-custom --weights yolov7-tiny.pt --freeze 50")
            else:
                os.system("python train.py --workers 1 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-tiny-custom --weights yolov7-tiny.pt --freeze 50")
    else:
        if(sel_model == '1'):
            print("Training the yolov7 model...")
            if(torch.cuda.is_available()):
                os.system("python train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt")
            else:
                os.system("python train.py --workers 1 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt")
        else:
            print("Training the yolov7-tiny model...")
            if(torch.cuda.is_available()):
                os.system("python train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-tiny-custom --weights yolov7-tiny.pt")
            else:
                os.system("python train.py --workers 1 --batch-size 8 --epochs 100 --img 640 640 --data data/GRIKDataset/data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-tiny-custom --weights yolov7-tiny.pt")
    print("Training is done.")

## 9-1. Test Model(macOS)
def test_model_mac(sel_model):
    if(sel_model == '1'):
        print("Testing the yolov7 model...")
        if(torch.backends.mps.is_available()):
            os.system("python detect.py --weights ./runs/train/yolov7-custom/weights/best.pt --device mps --conf 0.25 --img-size 640 --source ../test05.jpg")
        else:
            os.system("python detect.py --weights ./runs/train/yolov7-custom/weights/best.pt --conf 0.25 --img-size 640 --source ../test05.jpg")
    else:
        print("Testing the yolov7-tiny model...")
        if(torch.backends.mps.is_available()):
            os.system("python detect.py --weights ./runs/train/yolov7-tiny-custom/weights/best.pt --device mps --conf 0.25 --img-size 640 --source ../test05.jpg")
        else:
            os.system("python detect.py --weights ./runs/train/yolov7-tiny-custom/weights/best.pt --conf 0.25 --img-size 640 --source ../test05.jpg")
    print("Testing is done.")

## 9-2. Test Model(Windows or Linux)
def test_model(sel_model):
    if(sel_model == '1'):
        print("Testing the yolov7 model...")
        if(torch.cuda.is_available()):
            os.system("python detect.py --weights ./runs/train/yolov7-custom/weights/best.pt --device 0 --conf 0.25 --img-size 640 --source ../test05.jpg")
        else:
            os.system("python detect.py --weights ./runs/train/yolov7-custom/weights/best.pt --conf 0.25 --img-size 640 --source ../test05.jpg")
    else:
        print("Testing the yolov7-tiny model...")
        if(torch.backends.mps.is_available()):
            os.system("python detect.py --weights ./runs/train/yolov7-tiny-custom/weights/best.pt --device 0 --conf 0.25 --img-size 640 --source ../test05.jpg")
        else:
            os.system("python detect.py --weights ./runs/train/yolov7-tiny-custom/weights/best.pt --conf 0.25 --img-size 640 --source ../test05.jpg")
    print("Testing is done.")


if __name__ == '__main__':
    print("Start this program...")
    # Download the model
    download_yolov7()
    # Install required libraries
    install_requirements()
    # Check the current directory
    check_directory()
    # Check the current system
    check_system()
    # Check the platform
    this_platform = check_platform()
    # Unzip the dataset
    unzip_dataset(this_platform)
    # Select Model
    while(1):
        select_model = input("Select the model 1: yolov7 2: yolov7-tiny >>")
        if(select_model == '1' or select_model == '2'):
            break
        else:
            print("Please select 1 or 2.")
    # Download weight file
    download_weight_file(select_model)
    # Copy yaml file
    print("Copying the yaml file...")
    if(select_model == '1'):
        shutil.copyfile("../yaml/yolov7-custom.yaml", "cfg/training/yolov7-custom.yaml")
    else:
        shutil.copyfile("../yaml/yolov7-tiny-custom.yaml", "cfg/training/yolov7-tiny-custom.yaml")
    print("Yaml file is copied successfully.")
    # Train the model
    if(this_platform == "macOS"):
        train_model_mac(select_model)
    else:
        train_model(select_model)
    # Test the model
    if(this_platform == "macOS"):
        test_model_mac(select_model)
    else:
        test_model(select_model)

    # Copy the result weight file
    print("Copying the result weight file...")
    ## Check the save directory is already exist
    if not os.path.exists("../Training_Files"):
        os.makedirs("../Training_Files")
    ## Copy the weight file
    myWName = input("Enter file name to save >>")
    if(select_model == '1'):
        shutil.copyfile("./runs/train/yolov7-custom/weights/best.pt", f"../Training_Files/yolov7-custom-{myWName}.pt")
    else:
        shutil.copyfile("./runs/train/yolov7-tiny-custom/weights/best.pt", f"../Training_Files/yolov7-tiny-custom-{myWName}.pt")
    print("Weight file is copied successfully.")

    # Copy the result image
    ## Check the save directory is already exist
    if not os.path.exists("../Train_Test_Results"):
        os.makedirs("../Train_Test_Results")
    ## Copy the result image
    print("Copying the result image...")
    myName = input("Enter file name to save >>")
    shutil.copyfile("./runs/detect/exp/test05.jpg", f"../Train_Test_Results/{myName}.jpg")
    print("Result image is copied successfully.")

    # Delete yolov7 directory
    print("Deleting the yolov7 directory...")
    ## Change the directory
    print("Current Directory: ", os.getcwd())
    print("Changing the directory...")
    os.chdir("..")
    print("Directory is changed successfully.")
    ## Delete the directory
    print("Start deleting the yolov7 directory...")
    if(this_platform == "Windows"):
        os.system("rmdir /s yolov7")
    else:
        os.system("rm -rf yolov7")
    print("yolov7 directory is deleted successfully.")

    # Finish the program
    print("All processes are done.")