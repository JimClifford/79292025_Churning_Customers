# Telco Churn Prediction Project
## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Model Optimization](#model-optimization)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Requirements](#requirements)
- [Usage](#usage)



## Overview
This repository focuses on predicting Telco customer churn using a sample dataset. The project involves data extraction, feature extraction, training a functional Artificial Neural Network (ANN), optimization, and deployment using the Streamlit library.

## Dataset
This repository uses a dataset containing information about Telco customers. [Add a brief description of the dataset and its source.]

## Data Preprocessing
Data preprocessing involves cleaning and organizing the dataset to prepare it for feature extraction and model training.

## Feature Extraction
Feature extraction is performed to select the most important features in the dataset that contribute significantly to predicting churn - Uses the RandomClassifier library

## Model Training
A functional Artificial Neural Network (ANN) is trained on the preprocessed data to predict customer churn.

## Model Optimization
The initial model is optimized (using GridSearch) to enhance its performance, resulting in improved accuracy and AUC scores.

## Model Evaluation
The final model is evaluated, and accuracy and AUC scores are calculated. The initial scores were 0.82 and 0.74, and after optimization, they improved to 0.8255 and 0.76, respectively.

## Deployment
The trained and optimized model is saved(Using pickle), and a Python script using the Streamlit library is created for deploying the model as a web application.

## Link to Video Demonstration
https://drive.google.com/file/d/1C6EYFZ_FNELADs2-jWMe_K1qaXvV0fbf/view?usp=sharing

## Requirements
Ensure the following packages are installed in your Python environment:

- streamlit=>1.11.0

## Usage
Instructions for use - optimal option - Command line interface:

1. change the directory of the interface to folder where the churn.py script is 
2. use streamlit run churn.py to execute the deployment stage
3. on the webpage, click the deploy button to generate a unigue url address to the web app.






