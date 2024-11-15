# GRIK
Hanyang University Graduation Project in 2024.

## License
"This project is licensed under the terms of the GNU GPL-3.0 License." <br>
Please make sure this license when you fork our project.

## Motivation and Objectives
### Motivation
In engineering lectures, handwritten notes such as graphs and diagrams are just as important as the script. Existing lecture summarization services convert audio into text and then use LLMs for summarization. This approach has led to the issue of summaries not reflecting the handwritten notes.
### Objectives
The goal is to perform object detection on the handwritten notes and then process those sections to summarize them into a PDF.

## Technical contributions
### 1. Baseline
We established a baseline for our project by conducting object detection with YOLOv7-tiny, trained for 100 epochs with a batch size of 8. The Structural Similarity Index (SSIM) from SciKit was employed to assess the quality of the images. Background removal from the handwritten notes was implemented using a nested for loop.
### 2. Skill/Knowledge
#### 2.1 Dataset
Frames containing handwritten notes were extracted from certain lecture videos and club seminars. The handwritten sections were then bounded to create a custom dataset, which was developed using Roboflow.
#### 2.2 Model
Transfer learning was conducted using the pre-trained weight file of YOLOv7-tiny on the custom dataset. To achieve results comparable to those obtained with YOLOv7, adjustments were made to the epoch and batch size parameters. It was determined that an epoch of 400 and a batch size of 32 were optimal settings.
### 3. Novelty
By utilizing the SSIM implementation from OpenCV instead of SciKit, we achieved a twofold improvement in performance. The implementation of YOLOv7-tiny was optimized through TensorRT, resulting in a reduction of processing time by more than tenfold. In the background removal process, rather than merely subtracting the original image from the handwritten image, we employed the subtracted image as a mask, thereby reducing the processing time per frame.

## Implementation
The project leverages the computational power of the NVIDIA Jetson Nano board and its dedicated GPU to run deep learning models for object detection and speech recognition efficiently. The system is implemented using Python and various deep learning frameworks, such as TensorFlow and PyTorch.

## Key Features
### Object Detection
The system utilizes advanced object detection algorithms to identify and extract relevant visual information from the lecture videos, such as slides, diagrams, and handwritten notes. <br>
This visual data is then incorporated into the generated PDF notes for better comprehension and context.
### Summarization and Formatting
The transcribed text and extracted visual information are intelligently processed and summarized to create concise and well-strutured PDF notes. The notes are formatted with appropriate headings, bullet points, and visual aids, ensuring a clear and organized presentation of the lecture content.

## Potential Applications
### Educational Institutions
Lecture Summarizer can be a valuable tool for students and educators, enabling them to quickly review and revisit lecture content in a condensed and organized format.
### Online Learning Platforms
The system can be integrated into online learning platforms to provide learners with comprehensive and easily digestible summaries of video lectures.
### Corporate Training
Businesses can utilize Lecture Summarizer to create concise training materials from recorded sessions, facilitating knowledge transfer and employee development.

## About Our Project
### Why this project is named "GRIK"
We ofthen find that project code names are inspired by food items, such as Eclipse, Bread, and Honeycomb. <br>
In our case, we named our project "GRIK" because of our fondness for Greek yogurt. <br>
The name not only reflects our culinary preferences but also pays homage to the rich cultural heritage of Greece.
### Dependency
- Jetson Nano 4GB Dev. Board
- Python 3.6.9
### Requirements
```
Imagehash
numpy == 1.9.5
Cython < 3.0 -> Over 3.0, not support python 3.6
scikit-image -> SSIM-CPU version
PIL
reportlab
natsort
torch == 1.10.0
TensorRT == 8.2 -> If you install Official Jetpack version that support Jetson Nano, you don't need to install this.
collections
wget
```

## How to run our project
Step 1. Download our project file. <br>
Step 2. Open your terminal. <br>
Step 3. Run `python3 app.py`

## Implementation Mechanism

<div align="center">
 <img width="488" alt="image" src="https://github.com/user-attachments/assets/0156daa5-c8ce-442d-97f0-cf7ae7a0c54d">

  [ Fig 01. Program Architecture Diagram ]
</div>

This project processes video files through a total of four stages. Initially, the video file is received in `app.py`, which passes the video storage path as a parameter to `Imagehash.py`. Subsequently, images are extracted using the pHash-based Hamming Distance to differentiate between the pre-annotated and post-annotated frames. The image paths are then forwarded to `detection.py`, where YOLOv7-tiny is employed to extract the annotated sections. The extracted image paths are passed to extract.py for background removal. Finally, all processed images are compiled into a single PDF using `makePDF.py`.

### Step 1. Imagehash
Initially, the algorithm employed in this project was the Structural Similarity Index (SSIM). However, due to the excessive processing time encountered[1] during implementation on the Jetson Nano, we transitioned to using Imagehash for improved performance.
<div align="center">
  <img width="976" alt="image" src="https://github.com/user-attachments/assets/aaabb8a4-560a-4ca2-9b74-f1f10b1936f0">

  [ Fig. 02. Runtime Comparison of Similarity Algorithms ]
</div>

Comparing the results, the runtime for Scikit's SSIM was 411 seconds for a 720p video at 11 minutes and 24 frames, while OpenCV achieved a runtime of 200 seconds. In contrast, Imagehash demonstrated a significantly reduced runtime of 124 seconds. The substantial decrease in runtime upon switching to Imagehash indicates that the algorithmic improvement was meaningful and beneficial.

<div align="center">
 <img width="375" alt="image" src="https://github.com/user-attachments/assets/5466c5f1-6bc7-4717-a9ea-e720ac885d10">
  
 <img width="188" alt="image" src="https://github.com/user-attachments/assets/9e547f57-8eb0-4612-8ac0-47c4713e00b7">
 
 [ Fig. 03. pHash Computation Process and Method for Calculating Hamming Distance ]
</div>

By employing this method, the project utilized dynamic programming (DP) to calculate the Hamming distance, thereby reducing the time complexity compared to SSIM.

<div align="center">
 <img width="488" alt="image" src="https://github.com/user-attachments/assets/97b3598f-4abc-4a96-8d1e-6dbe1d6cb3c3">

[ Fig. 04. Algorithm Improvement through the Application of Dynamic Programming (DP) ]
</div>

### Step 2. Object Detection
To accurately recognize the annotations and extract the annotated sections, we determined that an Object Detection approach was appropriate. Consequently, we opted to use YOLOv7-tiny, which had been verified for feasibility on the Jetson Nano. However, a challenge arose as YOLOv7 was a model not trained on annotation data. To address this, we utilized the Roboflow platform to create a custom dataset by extracting frames from videos of seminars and external lectures. Subsequently, we conducted training in the Colab environment using hyperparameters.

<div align="center">
 <img width="976" alt="image" src="https://github.com/user-attachments/assets/9dcc90db-8624-4447-b2ec-7a15addd4003">

 [ Fig. 05. Training Results by Hyperparameters ]
</div>

The training results indicated a trend of increasing accuracy with larger values for both epoch and batch size. In the final version, we utilized a model trained with 400 epochs and a batch size of 32.

<div align="center">
 <img width="336" alt="image" src="https://github.com/user-attachments/assets/4c75aef7-b9b4-41c2-a59d-aebafc28e9f6">

 [ Fig. 06. Inference Time per Image Before Optimization ]
</div>

However, due to performance constraints on the Jetson Nano, the inference time per image was 28 seconds. This led us to conclude that GPU optimization was necessary. Consequently, we developed the following optimization algorithm.

<div align="center">
 <img width="488" alt="image" src="https://github.com/user-attachments/assets/aa762f11-0e77-421f-9d4b-54644abdf387">

 [ Fig. 07. Quantization and TensorRT Optimization Algorithm ]
</div>

Therefore, we chose to convert the model to ONNX format and apply TensorRT optimization. However, we encountered an issue with the 'NMS layer processing.' This layer is positioned before the output of YOLOv7, and the challenge arose because TensorRT does not support optimization for the NMS layer. To resolve this, we successfully adapted the method outlined in the official YOLOv7 GitHub repository for optimization.

<div align="center">
 <img width="340" alt="image" src="https://github.com/user-attachments/assets/bbf5f0c7-92fe-4219-b2c8-e912e8a02123">

 [ Fig. 08. Results After TensorRT Optimization ]
</div>

As a result, we were able to reduce the inference time from the original 28 seconds per image to 2.5 seconds. Notably, we attribute a significant portion of this improvement to the FP16 quantization implemented during the conversion to TensorRT.

### Step 3. Background Removal
Initially, we aimed to extract the annotations from the images using Absolute Differentiation, comparing the post-annotation images with the pre-annotation images. However, we encountered an issue where the original annotation colors were distorted in areas overlapping with the background. To address this, we devised a method that utilized the Absolute Differentiated images as masks, applying them back to the annotation images. During this process, we found that using NumPy syntax significantly reduced the processing time per image, leading us to adopt this approach.

<div align="center">
 <img width="470" alt="image" src="https://github.com/user-attachments/assets/6ad388f0-e53a-4930-85ac-6d02aa11f02e">

  [ Fig. 09. Background Removal Algorithm ]
</div>

### Step 4. Make PDF File
<div align="center">
 <img width="488" alt="image" src="https://github.com/user-attachments/assets/fa6fcca6-34c7-4d15-acb8-0b95cb98eb8a">

 [ Fig. 10. PDF Generation Method ]
</div>

In this step, we utilized the images generated in steps 1 to 3 to construct the frames, which included the annotated frame, the annotation itself, and the frame without annotations.

## Discussion and Conclusion
There are various metrics for calculating similarity. Among PSNR and SSIM, the SSIM method better reflects the differences perceived by the human eye. While SSIM is known to require less computational load than YOLOv7, we found that as image quality increases, the computational load grows exponentially. Although resizing images using OpenCV can reduce computational load, we anticipated that this resizing process would also increase latency. In this regard, choosing to use hashing to quantify structural differences was a good decision. Additionally, we discovered that optimizing the PyTorch model with TensorRT and using a masking method instead of nested loops can enhance computational speed.

## Citation
[1] Fitri N. Rahayu, ulrich Reiter, etc, “Analysis of SSIM Performance for Digital Cinema Applications”, IEEE 978-1-4244-4370-3/09, 2009

## BUILD History
- Jun 14, 2024: Object Detection available
- Jun 20, 2024: Change Similarity Algorithm **Before: SSIM, NOW: imagehash**

## Demo
You can see our demo video [here](https://drive.google.com/file/d/1F_RACJmRroySb1lgFV0WIAQB-g7R_o2E/view?usp=sharing)
