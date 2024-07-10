# HighRiskCancer
This repository contains a deep learning model for classifying high risk cancer patients. The training dataset have expression of genes and labels (1 for high risk and 0 for low risk). The model is implemented using TensorFlow and is trained on a dataset of 318 gene features.

## Contents
* `model_DL.pkl`: The trained TensorFlow model for classification.
* `model.py`: The script for training and using the model.
* `dataset/kaggle_train.csv`: Training data used for model training.
* `dataset/kaggle_test.csv`: Test data used for predictions.
* `dataset/sample.csv`: Sample CSV file for generating output predictions.

## Requirements
* pandas
* numpy
* scikit-learn
* tensorflow

You can install the required packages using pip:
```
pip install requirements.txt
```

## Command line arguments
* -o, --output: Name of the output CSV file to generate, will contain ID column and label probability (required).
* -m, --model: Name of the model to use, choose 'DL' for TensorFlow DeepLearning Classifier (required).
* -s, --save: Save the trained model to a file (optional).
* -l, --load: Load the trained model from a file (optional).

## Training and saving a New Model
```
python model.py -o Results.csv -m DL -s
```

## Load Existing Model and Make Predictions
```
python model.py -o Results.csv -m DL -l model_DL.pkl
```

## Data
The training and test data used in this project are provided in the dataset folder.

## Note
This project was part of a kaggle competition.


