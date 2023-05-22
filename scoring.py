"""
This script is to accomplish model scoring
date: May 22th 2023
"""
import logging
import pandas as pd
import pickle
import os
from sklearn import metrics
import json

# Initialize logging
logging.basicConfig(filename='logs/scoring.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

#################Load config.json and get path variables
def get_config():
    with open('config.json','r') as f:
        config = json.load(f) 

    model_path = os.path.join(config['output_model_path']) 
    test_data_path = os.path.join(config['test_data_path']) 
    return model_path, test_data_path


#################Function for model scoring
def score_model():
    """
    Function that accomplishes model scoring
    """
    #this function should take a trained model, load test data, 
    # and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    model_path, test_data_path = get_config()
    logging.info("Loading testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_test = test_df.pop('exited')
    X_test = test_df.drop(['corporation'], axis=1)

    # Load the model
    logging.info("Loading trained model")
    model = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))

    # Evaluate the model
    y_pred = model.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred)

    # Save the evaluation score into a file
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1_score}")
    logging.info("Evaluation scores saved")

if __name__ == '__main__':
    logging.info("Running scoring.py script")
    score_model()