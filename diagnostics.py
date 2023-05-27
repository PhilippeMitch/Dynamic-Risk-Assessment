"""
This script is to find problems - if any exist - in the model and data.
author: Philippe Jean Mith
date: May 23th 2023
"""
import os
import ast
import json
import pickle
import timeit
import logging
import subprocess
import pandas as pd

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
    """
    This function verify if there is any problem with the prediction
    Input
    -----
    dataset: DataFrame
        Data that will be used to make prediction
    Output
    ------
    y_pred: List
        A list of predictions
    """
    # Read the deployed model and a test dataset, calculate predictions
    _, test_df_path, deployment_path = get_config()

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

    X_test = dataset.drop(['corporation'], axis=1)
    # Evaluate the model
    y_pred = model.predict(X_test)
    # Return value should be a list containing all predictions
    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    """
    This function calculate the means, medians, and standard deviations 
    for each for each numeric column in the dataset
    """
    #calculate summary statistics here
    data_csv_path, _, _ = get_config()
    # load the dataset
    try:
        data = pd.read_csv(os.path.join(data_csv_path, 'finaldata.csv'))
    except FileNotFoundError as err:
        logging("Error: Could not found the finaldata.csv")
    # Get the numeric column
    # numeric_col = [col for col in data.columns if data[col].dtypes != "O"][:-1]
    stat_summary_df = data.describe()
    # lst_stat_summary = stat_summary_df.iloc[[1, 5, 2],:-1].values.tolist()
    stat_summary_dict = stat_summary_df.iloc[[1, 5, 2],:].to_dict('dict')
    for key, _ in stat_summary_dict.items():
        stat_summary_dict[key]['median'] = stat_summary_dict[key].pop('50%')
    #return value should be a list containing all summary statistics
    return stat_summary_dict

##################Function to get timings
def execution_time():
    """
    This function  times how long it takes to perform 
    the data ingestion and the training
    Output
    ------
    timing: dict
        A dictionary that contains the timing for 
        the ingest.py and training.py
    """
    timing = {}
    # Calculate timing of ingestion.py
    logging.info("Calculate timing of ingestion.py")
    ingest_starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing['ingestion_timing']=timeit.default_timer() - ingest_starttime
    # Calculate timing of training.py
    logging.info("Calculate timing of training.py")
    train_starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing['training_timing']=timeit.default_timer() - train_starttime
    #return a dict of 2 timing values in seconds
    return timing

##################Function to check dependencies
def outdated_packages_list():
    """
    This function is to verify if the modules imported are up-to-date.
    """
    #get a list of
    logging.info("Cheking the dependencies")
    libraries = subprocess.check_output(['pip', 'list', '--outdated', '--format', 'json'])
    libraries_dict = ast.literal_eval(libraries.decode('utf-8'))
    # Remove the latest_filetype column 
    for item in libraries_dict:
        del item['latest_filetype']
    
    return libraries_dict

def missing_data():
    """
    This function calculate the percent of each column that
    consists of NA values
    """
    data_csv_path, _, _ = get_config()
    # load the dataset
    try:
        data = pd.read_csv(os.path.join(data_csv_path, 'finaldata.csv'))
    except FileNotFoundError as err:
        logging("Error: Could not found the finaldata.csv")

    logging.info("Calculating missing data")
    missing_list = {
        col: {'percent ': perc} for col, perc in zip(
                            data.columns, 
                            data.isna().sum() / data.shape[0] * 100
        )
    }

    return missing_list

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
