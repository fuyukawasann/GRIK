# GRIK
Hanyang University Graduation Project in 2024.

## License
"This project is licensed under the terms of the GNU GPL-3.0 License." <br>
Please make sure this license when you fork our project.

## Introduction
This project aims to develop an intelligent system that can automatically summarize video lectures into concise PDF notes, leveraging the power of object detection and voice-to-text technologies. The system is designed to run on the NVIDIA Jetson Nano board, a compact yet powerful platform for deploying deep learning models.

## Key Features
### Object Detection
The system utilizes advanced object detection algorithms to identify and extract relevant visual information from the lecture videos, such as slides, diagrams, and handwritten notes. <br>
This visual data is then incorporated into the generated PDF notes for better comprehension and context.
### Voice-to-Text Transcription
Lecture audio is transcribed into text using state-of-the-art speech recognition models. The transcribed text serves as the primary source for generating the textual content of the PDF notes.
### Summarization and Formatting
The transcribed text and extracted visual information are intelligently processed and summarized to create concise and well-strutured PDF notes. The notes are formatted with appropriate headings, bullet points, and visual aids, ensuring a clear and organized presentation of the lecture content.

## Implementation
The project leverages the computational power of the NVIDIA Jetson Nano board and its dedicated GPU to run deep learning models for object detection and speech recognition efficiently. The system is implemented using Python and various deep learning frameworks, such as TensorFlow and PyTorch.

## Potential Applications
### Educational Institutions
Lecture Summarizer can be a valuable tool for students and educators, enabling them to quickly review and revisit lecture content in a condensed and organized format.
### Online Learning Platforms
The system can be integrated into online learning platforms to provide learners with comprehensive and easily digestible summaries of video lectures.
### Corporate Training
Businesses can utilize Lecture Summarizer to create concise training materials from recorded sessions, facilitating knowledge transfer and employee development.

## Prepare
1. Jetson Nano Board
2. Micro SD Card
3. Webcam

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
OpenCV == 4.6.0 with CUDA Acceleration
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
**We use QualitySSIM in CV2, thus you must install OpenCV-contrib version** <br>
**This takes at least 3 hours**

## How to run our project
Step 1. Download our project file. <br>
Step 2. Open your terminal. <br>
Step 3. Run `python3 app.py` <br>

## BUILD History
- Jun 14, 2024: Object Detection available
