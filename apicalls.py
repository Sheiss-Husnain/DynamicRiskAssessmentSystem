import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json','r') as f:
    config = json.load(f)


def responses(URL):
    response1 = requests.post(URL+'/prediction?input_data=testdata/testdata.csv').content
    response2 = requests.get(URL+'/scoring').content
    response3 = requests.get(URL+'/summarystats').content
    response4 = requests.get(URL+'/diagnostics').content

    return [response1, response2, response3, response4]

def api_write(response_all):
    with open(os.path.join(os.getcwd(),config['output_model_path'],'apireturns2.txt'),'w') as file:
#         response_all = responses(URL)
        file.write(str(response_all))

if __name__ == '__main__':
    resp = responses(URL)
    api_write(resp)



