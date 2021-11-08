
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

model_path=os.path.join(config['prod_deployment_path'])
##################Function to get model predictions
def model_predictions(model_path, test_data_path):
    #read the deployed model and a test dataset, calculate predictions
    
    model = pickle.load(open(os.path.join(os.getcwd(), model_path, "trainedmodel.pkl"),'rb'))
    
    test_data = pd.read_csv(os.path.join(os.getcwd(),test_data_path,'testdata.csv'))
    
    X_test = test_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1,3)
    y_test = test_data["exited"].values.reshape(-1,1).ravel()
                            
    y_pred = list(model.predict(X_test))
    return y_pred #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
                            
    data = pd.read_csv(os.path.join(os.getcwd(),dataset_csv_path,'finaldata.csv'))
        
    stats=[]                            
    for col in cols:
        stats_ = []
        stats_.append(np.mean(data[col]))
        stats_.append(np.median(data[col]))
        stats_.append(np.std(data[col]))                            

        stats.append(stats_)
    return stats#return value should be a list containing all summary statistics

def missing_data():
    data = pd.read_csv(os.path.join(os.getcwd(),dataset_csv_path,'finaldata.csv'))
    percent_na = list(data.isna().sum(axis=1)/data.shape[0])
    return percent_na
                            
##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    times = []
    for file in ['ingestion.py','training.py']:
        starttime=timeit.default_timer()
        response = subprocess.run(['python3',file])
        timing = timeit.default_timer()-starttime
        times.append(timing)
    return times#return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdate = subprocess.check_output(['pip','list','--outdated'])
                            
    with open(os.path.join(os.getcwd(),'terminal_output.txt'),'wb') as file:
        file.write(outdate)
                            
    with open(os.path.join(os.getcwd(),'terminal_output.txt'),'rb') as file:
        packages = file.read()

    packages = packages.decode('utf-8').split('\n')
                            
    cols = packages[0].split()
    cols = [col for col in cols if col!=''][:3]
    
    df = pd.DataFrame(columns=cols)
                            
    for i in range(2,len(packages)):
        row = packages[i].split()
        row = [row for row in row if row !=''][:3]
                            
    if len(row)==3:
        row_ = pd.Series(row,index=df.columns)
        df = df.append(row_,ignore_index=True)
    return df

if __name__ == '__main__':
    model_predictions(model_path, test_data_path)
    dataframe_summary()
    execution_time()
    outdated_packages_list()
    missing_data()





    
