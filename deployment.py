"""
This script is to accomplish model deployment
author: Philippe Jean Mith
date: May 22th 2023
"""
import logging
import shutil
import os
import json

# Initialize logging
logging.basicConfig(filename='logs/deployment.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

##################Load config.json and correct path variable
def get_config():
    with open('config.json','r') as f:
        config = json.load(f) 

    dataset_csv_path = os.path.join(config['output_folder_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    model_path = os.path.join(config['output_model_path']) 
    return dataset_csv_path, prod_deployment_path, model_path


####################function for deployment
def store_model_into_pickle():
    """
    Function that accomplishes the deployment
    """
    #copy the latest pickle file, the latestscore.txt value, 
    # and the ingestfiles.txt file into the deployment directory
    data_path, deployment_path, model_path = get_config()
    logging.info("Deploying trained model to production")
    os.makedirs(deployment_path, exist_ok=True)

    # Copy the ingestedfiles.txt to the deployment path
    try:
        shutil.copy(os.path.join(data_path, 'ingestedfiles.txt'), deployment_path)
    except FileNotFoundError as err:
        logging.error("Error: The ingestedfiles.txt is not found")

    # Copy the trainedmodel.pkl file to the deployment path
    for file in os.listdir(model_path):
        try:
            shutil.copy(os.path.join(model_path, file), deployment_path)
        except FileNotFoundError as err:
            logging.error(f"Error: Could not found the {file} file")

    logging.info("File copied to the deployment path")

if __name__ == '__main__':
    logging.info("Running deployment.py script")
    store_model_into_pickle()