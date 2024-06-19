import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.utils import save_file
import logging
import pickle as pkl
import os
logging.basicConfig(level=logging.INFO)

def change_datatypes(train_data,test_data):
    logging.info("Change the datatypes of TotalCharges col of test data")
    test_data['TotalCharges']=test_data['TotalCharges'].replace(" ",np.nan)
    test_data['TotalCharges']=test_data['TotalCharges'].astype("float64")

    logging.info("Change the datatypes of TotalCharges col of train data")
    train_data['TotalCharges']=train_data['TotalCharges'].replace(" ",np.nan)
    train_data['TotalCharges']=train_data['TotalCharges'].astype("float64")

    return train_data,test_data

def inisiate_data_transformation(train_path,test_path):
    # set the path for saving process data
    # train_process_path=os.path.join("Data/process","train_process.csv")
    # test_process_path=os.path.join("Data/process","test_process.csv")
    processor_path=os.path.join("models","processor.pkl")

    train_df=pd.read_csv(train_path)
    test_df=pd.read_csv(test_path)

    # Data Contain 
    # Total charges dtype is object change thee dtypes
    train_df,test_df=change_datatypes(train_data=train_df,test_data=test_df)
    logging.info("Datatypes Handle Successfully!")


    logging.info("Saperate feature and lable")
    x_train=train_df.drop(columns=['Churn'])
    y_train=train_df['Churn']
    x_test=test_df.drop(columns=['Churn'])
    y_test=test_df['Churn']
    
    
    logging.info("Saperate numerical and categorical columns")
    num_col=x_train.select_dtypes("number").columns
    cat_col=x_train.select_dtypes("object").columns
    logging.info(f"numerical col {num_col} categorical columns {cat_col}")

    logging.info("Building pipeline for preprocessing")

    num_pipe=Pipeline(steps=[
        ("impute",SimpleImputer(strategy="median")),
        ("scale",StandardScaler())
    ])

    cat_pipe=Pipeline(steps=[
        ("Impute",SimpleImputer(strategy="most_frequent")),
        ("Encode",OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'))
    ])
    
    logging.info("Building Transformer")
    processor=ColumnTransformer(transformers=[
        ("Num_transformation",num_pipe,num_col),
        ("Cat_transform",cat_pipe,cat_col)
    ],remainder="passthrough")

    logging.info("Tranform train data")
    x_train_transform=processor.fit_transform(x_train)
    logging.info("Tranform test data")
    x_test_transform=processor.transform(x_test)
    
    save_file(file_path=processor_path,obj=processor)
    logging.info(f"Processor save in this location {processor_path}")

    logging.info("Combine feature and label")
    train_array=np.c_[
        x_train_transform,np.array(y_train)
    ]

    test_array=np.c_[
        x_test_transform,np.array(y_test)
    ]
    
    return[
        train_array,
        test_array,
        processor_path
    ]
 
    
    

# if __name__=="__main__":
#     inisiate_data_transformation('Data/raw/train.csv','Data/raw/test.csv')