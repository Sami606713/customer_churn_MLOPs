import pandas as pd
import pickle as pkl
import os
import logging
logging.basicConfig(level=logging.INFO)

def get_data():
    try:
        logging.info("Reading the data from url")
        # URL of the dataset
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'

        # Load the dataset
        data = pd.read_csv(url)

        # data.head()

        return data
    except Exception as e:
        return e

def save_file(file_path,obj):
    with open(file_path,"wb")as f:
        pkl.dump(obj=obj,file=f)
