import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    #check for datasets, compile them together, and write to an output file
    

    
    #######################
#     if not os.path.exists(os.path.join(os.getcwd(), output_folder_path)):
#         os.makedirs(os.path.join(os.getcwd(), output_folder_path))
        
#     df = pd.DataFrame(columns=['corporation','lastmonth_activity','lastyear_activity','number_of_employees','exited'])
    
#     filenames = os.listdir(os.getcwd()+'/'+input_folder_path)
                      
#     with open(os.path.join(output_folder_path,"ingestedfiles.txt"),'w') as f:
#         for file in filenames:
#             df1 = pd.read_csv(os.getcwd()+'/'+input_folder_path+'/'+file)
#             df = df.append(df1)
#             f.write(file+'\n')

#     df = df.drop_duplicates()
#     df.to_csv(os.getcwd()+'/'+output_folder_path+'/'+'finaldata.csv',index=False)
    
#     return df
              
    
    dir = os.path.join(os.getcwd(), input_folder_path)

    df = pd.DataFrame(columns=['corporation','lastmonth_activity','lastyear_activity','number_of_employees','exited'])
    
    filenames = os.listdir(dir)
    
    for file in filenames:
            df1 = pd.read_csv(os.path.join(dir,file))
            df1.drop_duplicates(inplace=True)
            df = df.append(df1,ignore_index=True)
            
    output_filepath = os.path.join(os.getcwd(), output_folder_path)
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
        
    with open(os.path.join(output_folder_path,'ingestedfiles.txt'),'w') as file:
        for f in filenames:
            file.write(f+'\n')
    output_filepath = os.path.join(output_filepath, "finaldata.csv")
    df.to_csv(output_filepath, index=False)
    #make sure to add 'ingesteddata' folder   
    return df
if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, output_folder_path)
