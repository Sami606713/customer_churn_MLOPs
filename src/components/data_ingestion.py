from sklearn.model_selection import train_test_split
from src.utils import get_data
import os
import logging
logging.basicConfig(level=logging.INFO)

def inisiate_data_ingestion():
    # set the raw test and train data path
    raw_path=os.path.join("Data/raw","raw.csv")
    train_path=os.path.join("Data/raw","train.csv")
    test_path=os.path.join("Data/raw","test.csv")

    logging.info("getting the data")
    data=get_data()

    # drop the Customer ID
    data.drop(columns=["customerID"],inplace=True)
    logging.info(f"Saving data in this path {raw_path}")
    data.to_csv(raw_path,index=False)
    logging.info("Raw Data Save Successfully")

    logging.info("Splitting the data")
    train_data,test_data=train_test_split(data,test_size=0.2,random_state=43)

    logging.info(f'Save the training data in this path {train_path}')
    train_data.to_csv(train_path,index=False)

    logging.info(f'Save the testing data in this path {test_path}')
    test_data.to_csv(test_path,index=False)

    return [
        train_path,
        test_path
    ]
    




    