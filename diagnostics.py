
import pandas as pd
import logging
import pickle
import numpy as np
import timeit
import os
import json

# Initialize logging
logging.basicConfig(filename='logs/diagnostics.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

##################Load config.json and get environment variables
def get_config():
    with open('config.json','r') as f:
        config = json.load(f) 

    data_csv_path = os.path.join(config['output_folder_path'])
    test_df_path = os.path.join(config['test_data_path'])
    deployment_path = os.path.join(config['prod_deployment_path'])
    return data_csv_path, test_df_path, deployment_path

##################Function to get model predictions
def model_predictions(dataset=None):
    #read the deployed model and a test dataset, calculate predictions
    data_csv_path, test_df_path, deployment_path = get_config()

    if dataset is None:
        try:   
            dataset = pd.read_csv(os.path.join(test_df_path, 'testdata.csv'))
        except FileNotFoundError as err:
            logging.error("Error: Could not found the testdata.csv")
    
    try:
        # collect deployed model
        with open(os.path.join(deployment_path, 'trainedmodel.pkl'), 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError as err:
        logging.error("Could not found the trainedmodel.pkl file")

    y_test = dataset.pop('exited')
    X_test = dataset.drop(['corporation'], axis=1)
    # Evaluate the model
    y_pred = model.predict(X_test)
    # Return value should be a list containing all predictions
    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    data_csv_path, test_df_path, deployment_path = get_config()
    # load the dataset
    data = pd.read_csv(os.path.join(data_csv_path, 'finaldata.csv'))
    # Get the numeric column
    numeric_col = [col for col in data.columns if data[col].dtypes != "O"][:-1]
    stat_summary_df = data.describe()
    lst_stat_summary = stat_summary_df.iloc[[1, 5, 2],:-1].values.tolist()
    #return value should be a list containing all summary statistics
    return lst_stat_summary

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
