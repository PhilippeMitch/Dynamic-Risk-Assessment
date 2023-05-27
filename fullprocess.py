"""
This script is to Process automation of the ML model scoring, 
monitoring, and re-deployment process.
author: Philippe Jean Mith
date: May 23th 2023
"""
import json
import os
import re
import logging
import ingestion
import pandas as pd
import training
import subprocess
import deployment
import diagnostics
import reporting
from sklearn import metrics

logging.basicConfig(filename='logs/process_automation.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

###############Load config.json and get path variables
def get_config():
    with open('config.json','r') as f:
        config = json.load(f) 
    dataset_csv_path = os.path.join(config['output_folder_path'])
    test_data_path = os.path.join(config['test_data_path'])
    prod_path = os.path.join(config['prod_deployment_path'])
    input_data_path = os.path.join(config['input_folder_path'])
    output_model_path = os.path.join(config['output_model_path'])
    return dataset_csv_path, test_data_path, prod_path, input_data_path, output_model_path

##################Deciding whether to proceed, part 1
def check_model_drift():
    logging.info("Checking for model drift")
    #if you found new data, you should proceed. 
    # otherwise, do end the process here
    dataset_csv_path, _, prod_path, input_data_path, _ = get_config()
    with open(os.path.join(prod_path, 'latestscore.txt'), 'r') as file:
        f1_score = file.readline()
    deployed_f1_score = round(float(re.findall(r'\d\d*\.?\d+',f1_score)[0]), 2)
    data = pd.read_csv(os.path.join(dataset_csv_path,'finaldata.csv'))
    y_data = data.pop('exited')
    X_data = data.drop(['corporation'], axis=1)
    # Evaluate the model
    y_pred = diagnostics.model_predictions(data)
    new_f1_score = metrics.f1_score(y_data, y_pred)
    ##################Checking for model drift
    # check whether the score from the deployed model is different 
    # from the score from the model that uses the newest ingested data
    if round(new_f1_score, 2) >= deployed_f1_score:
        logging.info(f"There is no model drift New score: {new_f1_score}, \
                     old score: {deployed_f1_score}")
        return None
    else:
        ##################Deciding whether to proceed, part 2
        #if you found model drift, you should proceed. otherwise, do end the process here
        logging.info("Retrained the model")
        training.train_model()
        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        logging.info("Deploying the new model")
        deployment.store_model_into_pickle()
        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model
        logging.info("Running the system diagnostics and reporting")
        reporting.score_model()
        subprocess.run(['python', 'apicalls.py'])

##################Check and read new data
def check_new_data():
    logging.info("Checking for new data")
    #first, read ingestedfiles.txt
    _, _, prod_path, input_data_path, _ = get_config()
    with open(os.path.join(prod_path,'ingestedfiles.txt'), 'r') as file:
        ingested_files = {line.strip('\n') for line in file.readlines()}
    ingested_files = set([file.split(" ")[2].split("/")[1] for file in ingested_files])
    # Get the data in sourcedata folder
    source_files = set([file for file in os.listdir(input_data_path) if file.endswith('.csv')])
    #second, determine whether the source data folder 
    # has files that aren't listed in ingestedfiles.txt
    if len(source_files.difference(ingested_files)) == 0:
        logging.info("There is no new data found")
        return None
    else:
        logging.info("Ingesting new data")
        ingestion.merge_multiple_dataframe()
        check_model_drift()

if __name__ == '__main__':
    check_new_data()