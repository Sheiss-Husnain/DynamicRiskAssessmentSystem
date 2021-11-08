from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config_production.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
test_data = pd.read_csv(os.path.join(os.getcwd(),test_data_path,'testdata.csv'))
model_path = os.path.join(config['output_model_path'])
#################Function for model scoring
def score_model(model_path, test_data):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
#     test_data = pd.read_csv(os.path.join(os.getcwd(),test_data_path,'testdata.csv'))
    
    X_test = test_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1,3)
    y_test = test_data["exited"].values.reshape(-1,1).ravel()
                            
#     model_path = os.path.join(config['output_model_path'])
                            
    model = pickle.load(open(os.path.join(os.getcwd(),model_path,'trainedmodel.pkl'),'rb'))
                            
    y_pred = model.predict(X_test)
                            
    f1score = metrics.f1_score(y_test, y_pred)
                            
    with open(os.path.join(os.getcwd(),model_path,'latestscore.txt'),'w') as file:
        file.write('F1 Score: '+str(f1score))
    return f1score        
if __name__ == "__main__":
    score_model(model_path, test_data)                         