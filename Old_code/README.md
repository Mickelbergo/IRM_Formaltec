# 2D wound segmentation and classification in forensic medicine

![img](images/example_image.png)

## About
This repository contains the training code for the semantic segmentation of wounds on forensic images taken during the medical exams of injured persons. 
The code was used to train the models in the paper "Segmentation and classification of seven common wounds in forensic medicine". 

## Training
To run the code clone the repository and run main.py. Without changing the parameteres the model trained is a Se-ResNext-50 Feature Pyramid Model trained with a weighted BCE loss function. Change the parameters in config.py to adjust the training pipeline. 

## Required packages
<img src="https://img.shields.io/badge/-PyTorch-orange?style=for-the-badge&logo=appveyor"><img/>
<img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"><img/>
<img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/json-f0dd67?style=for-the-badge&logo=json&logoColor=black"/>
<img src="https://img.shields.io/badge/tqdm-0998eb?style=for-the-badge&logo=tqdm"/>
<img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black"/>
<img src="https://img.shields.io/badge/os-677075?style=for-the-badge"/>
<img src="https://img.shields.io/badge/sys-3d4042?style=for-the-badge"/>
<img src="https://img.shields.io/badge/-segmentation__models__pytorch-ff69b4?style=for-the-badge&logo"/>
<img src="https://img.shields.io/badge/-Seaborn-red?style=for-the-badge&logo"/>
