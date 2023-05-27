"""
This script is used to call the Flask API
author: Philippe Jean Mith
date: May 27th 2023
"""
import os
import json
import requests
import logging

# Initialize logging
logging.basicConfig(filename='logs/client_api.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"


with open('config.json','r') as f:
    config = json.load(f) 
test_data_path = os.path.join(config['test_data_path'])

#Call each API endpoint and store the responses
#put an API call here
try:
    test_data = os.path.join(test_data_path, 'testdata.csv')
    model_prediction = requests.post(f'{URL}:8000/prediction?dataset_file_path={test_data}').content
    model_prediction = model_prediction.decode().strip().replace('\n','')
    logging.info("The prediction endpoint was executed successfully")
except ValueError as err:
    logging.error("Could not get the prediction from the endpoint")
#put an API call here
try:
    score_prediction = requests.get(f'{URL}:8000/scoring').text
    logging.info("The score endpoint was executed successfully")
except ConnectionRefusedError as err:
    logging.error("Could not be able to connect")
#put an API call here
try:
    stat_sum = requests.get(f'{URL}:8000/summarystats').text
    logging.info("The stats endpoint was executed successfully")
except ConnectionRefusedError as err:
    logging.error("Could not be able to connect")
#put an API call here
try:
    diagnostics = requests.get(f'{URL}:8000/diagnostics').text
    logging.info("The diagnostics endpoint was executed successfully")
except ConnectionRefusedError as err:
    logging.error("Could not be able to connect")

#combine all API responses
responses = {
    "Model Prediction": model_prediction,
    "F1 Score": score_prediction,
    "Summary statistics": stat_sum,
    "System diagnostic": diagnostics

}

# for key, item in responses.items():
#     print(key, item)

with open('report/apireturns2.txt', 'w') as file:
    for key, item in responses.items():
        if type(item) is dict:
             for item_ket, item_value in item.items():
                file.write(f'\n\t{item_ket}\n\n\t{item_value}')
        else:
            file.write(f'\n\n{key}\n\n {item}')
#write the responses to your workspace