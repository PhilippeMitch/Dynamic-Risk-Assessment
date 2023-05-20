"""
This script is to make that data ingestion go smoothly and easily.
author: Philippe Jean Mith
date: May 20th 2022
"""

# Import the necessary libraries
import os
import glob
import json
import logging
import pandas as pd
from datetime import datetime

# Initialize logging
logging.basicConfig(filename='logs/process.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Load config.json and get input and output paths
def get_config():
    logging.info(f"INFO: Load the config file {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
    try:
        with open('config.json','r') as f:
            config = json.load(f) 
    except FileNotFoundError as err:
        logging.error(f"ERROR: the config.json file is not found {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
        raise err

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']

    return input_folder_path, output_folder_path

# Function for data ingestion
def merge_multiple_dataframe():
    """
    Function for Data ingestion process
    """
    input_folder_path, output_folder_path = get_config()
    os.makedirs(output_folder_path, exist_ok=True)
    #check for datasets
    logging.info(f"Info check for the datasets {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
    files_path = glob.glob(input_folder_path + "/*.csv")
    # compile them together
    df = pd.DataFrame()
    if len(files_path) == 0:
        logging.error(
            f"ERROR: There is no csv file in the {input_folder_path} \
                folder {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
        return f"No csv file found in the {input_folder_path} folder"
    df = pd.concat(map(pd.read_csv, files_path), ignore_index=True)
    # remove duplicates
    df = df.drop_duplicates(ignore_index=True)
    # write to an output file
    df.to_csv(os.path.join(output_folder_path, "finaldata.csv"))
    logging.info(f"Final dataset saved in the folder {output_folder_path} \
                 {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
    # saving a record of the ingestion
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as records:
        for file in files_path:
            records.write(f'Ingestion of {file} at : {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
    records.close()
    logging.info(f"Save metadata for Data ingestion {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")

if __name__ == '__main__':
    logging.info("Running ingestion.py {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")
    merge_multiple_dataframe()
