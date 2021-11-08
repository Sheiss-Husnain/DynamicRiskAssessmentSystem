from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle



import json
import os

from diagnostics import dataframe_summary, missing_data, execution_time

from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = pickle.load(open(os.path.join(os.getcwd(),config['prod_deployment_path'],'trainedmodel.pkl'),'rb'))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    input_data_path = request.args.get('input_data')
    input_data = pd.read_csv(input_data_path)
    
    X = input_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1,3)
    
    y_pred = prediction_model.predict(X).tolist()
    
    return str(y_pred)
    
#     return #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    model_path = os.path.join(config['output_model_path'])
    
    test_data_path = os.path.join(config['test_data_path'])
    
    test_data = pd.read_csv(os.path.join(os.getcwd(),test_data_path,'testdata.csv'))
    
    f1_score = score_model(model_path, test_data)
    
    return str(f1_score)#add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    stats_ = dataframe_summary()
    return str(stats_) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    timing = execution_time()
    percent_na = missing_data()
    
    return str((timing, percent_na))

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
