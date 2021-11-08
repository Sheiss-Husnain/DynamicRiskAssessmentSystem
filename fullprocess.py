# import training
# import scoring
# import deployment
import diagnostics
import apicalls
# import reporting
from apicalls import responses, api_write
from ingestion import merge_multiple_dataframe
from deployment import store_model_into_pickle
from scoring import score_model
from training import train_model
from reporting import report_confusion_matrix#score_model
from diagnostics import model_predictions, execution_time, dataframe_summary, outdated_packages_list
from numpy import double
import pandas as pd
import json
import os

import subprocess
import sys

with open("config_production.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path']) 
dataset_csv_path = os.path.join(config['output_folder_path']) 



ingested_files =[]
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as file:
    for line in file:
        ingested_files.append(line.rstrip())

data_flag = False
for filename in os.listdir(input_folder_path):
    if input_folder_path + "/" + filename not in ingested_files:
        data_flag = True

if not data_flag:
    print("New data was not ingested, add more to continue")
    exit(0)

merge_multiple_dataframe(config['input_folder_path'], config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
testdata = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))
score_model(model_path,testdata)

with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as file:
    f1_old = float(file.read().split()[2])

with open(os.path.join(model_path, "latestscore.txt"), "r") as file:
    f1_new = float(file.read().split()[2])

if f1_old <= f1_new:
    print("New F1 (%s) >= Old F1 (%s).  No drift detected." % (f1_new, f1_old))    
    exit(0)
else:
    print("New F1 (%s) < Old F1 (%s).  Drift detected therefore retraining model" % (f1_new, f1_old)) 
    
new_model = train_model(dataset_csv_path,model_path)

store_model_into_pickle(config['prod_deployment_path'], new_model, os.path.join(config['output_folder_path']))

model_predictions(model_path,test_data_path)

report_confusion_matrix(model_path,plot_path,test_data_path)

execution_time()
dataframe_summary()
missing_data()
outdated_packages_list()


