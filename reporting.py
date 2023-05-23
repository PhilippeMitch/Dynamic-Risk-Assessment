"""
This script is to create reports related to your ML model
author: Philippe Jean Mith
date: May 23th 2023
"""
import pickle
import logging
from diagnostics import model_predictions
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Initialize logging
logging.basicConfig(filename='logs/reporting.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

###############Load config.json and get path variables
def get_config():
    with open('config.json','r') as f:
        config = json.load(f) 

    dataset_csv_path = os.path.join(config['output_folder_path'])
    test_data_path = os.path.join(config['test_data_path'])

    return dataset_csv_path, test_data_path

##############Function for reporting
def score_model():
    # Calculate a confusion matrix using the test data and the deployed model
    dataset_csv_path, test_data_path = get_config()
    try:   
        dataset = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    except FileNotFoundError as err:
        logging.error("Error: Could not found the testdata.csv")
    y_test = dataset.pop('exited')
    X_test = dataset.drop(['corporation'], axis=1)
    y_pred = model_predictions(dataset)
    confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
    # Write the confusion matrix to the workspace


if __name__ == '__main__':
    score_model()
