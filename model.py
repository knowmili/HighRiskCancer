import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import joblib

parser = argparse.ArgumentParser()
'''Command line arguments using argparse'''
parser.add_argument("-o", "--output", required=True, help="Name of the Output CSV file to generate, will contain ID column and label probability")
parser.add_argument("-m", "--model", required=True, help="Name of the model to use, choose from 'DL150' for Tensorflow DeepLearning Classifier with 150 selected features, 'DL' for Tensorflow DeepLearning Classifier")
parser.add_argument("-s", "--save", action='store_true', help="Save the trained model to a file")
parser.add_argument("-l", "--load", help="Load the trained model from a file")

'''Assigning command line arguments using argparse to python variables'''
args = parser.parse_args()
output_file = args.output
model_option = args.model
save_model = args.save
load_model = args.load

print("All command line options provided:")
for key, value in args.__dict__.items():
    print(f'{key} : {value}')

if load_model:
    model = joblib.load(load_model)

else:
    if args.model == "DL":
        '''Loading the dataset'''
        # Load the training, test data, and sample CSV
        train_dataset = pd.read_csv("dataset/kaggle_train.csv")
        test_dataset = pd.read_csv("dataset/kaggle_test.csv")
        sample = pd.read_csv("dataset/sample.csv")

        '''Removing the label and ID column and another useless column and standardizing the dataset'''
        # Remove the 'Labels' column from the training data and 'PRG3' column
        train_labels = train_dataset.pop("Labels")
        train_dataset.drop("PRG3", axis=1, inplace=True)

        # Remove the 'PRG3' and 'ID' columns from the test data
        test_dataset.drop(["PRG3", "ID"], axis=1, inplace=True)

        # Standardize the datasets using StandardScaler
        scaler = StandardScaler()
        train_dataset_s = scaler.fit_transform(train_dataset)
        train_dataset = pd.DataFrame(train_dataset_s, columns=train_dataset.columns, index=train_dataset.index)
        test_dataset_s = scaler.transform(test_dataset)
        test_dataset = pd.DataFrame(test_dataset_s, columns=test_dataset.columns, index=test_dataset.index)

        '''Training the model using TensorFlow'''
        # Create a neural network model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(317)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(512, activation='linear'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(256, activation='linear'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(1, activation='linear'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        # Compile the model with optimizer, loss function, and metrics
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=['acc'])

        # Train the model with training data
        model.fit(np.asarray(train_dataset).astype('float64'), np.asarray(train_labels).astype('float64'), epochs=100)

        '''Predicting on the test dataset'''
        # Use the trained model to make predictions on the test data
        prob = model.predict(test_dataset)

        # Save the model if requested
        if save_model:
            joblib.dump(model, 'model_DL.pkl')

        # Update the 'Labels' column in the sample CSV with the predictions
        sample["Labels"] = prob

        # Save the results to a new CSV file
        sample.to_csv("Results.csv", index=False)

    else:
        print("Please select a valid option")