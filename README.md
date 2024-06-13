# GRIK
Hanyang University Graduation Project in 2024.

## License
"This project is licensed under the terms of the GNU GPL-3.0 License."
Please make sure this license when you fork our project.

## Goal of our project
We target an application that makes PDF files containing Lecture Notes and Writings.
We use Object Detection and voice-to-text to make summary notes at Jetson Nano Board.

## Prepare
1. Jetson Nano Board
2. Micro SD Card
3. Webcam

## About Our Project
### Why this project is named "GRIK"
We are interested in almost all project code names food as Eclipse, Bread, Honeycomb, etc.
At this point, we named "GRIK" because I like GRIK Yogurt :D
### Dependency
- Jetson Nano 4GB Dev. Board
- Python 3.6.9
### Requirements
```
OpenCV == 4.6.0 with CUDA Acceleration
numpy == 1.9.5
Cython < 3.0 **Over 3.0, not support python 3.6**
scikit-image
PIL
reportlab
natsort
torch == 1.10.0
TensorRT == 8.2
collections
wget
```
