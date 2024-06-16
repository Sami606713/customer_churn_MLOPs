import pandas as pd
import pickle as pkl
import mlflow
import mlflow.sklearn
import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
import os
import logging
logging.basicConfig(level=logging.INFO)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


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

def save_report(file_path,report):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "w") as f:
        json.dump(report, f, cls=NumpyEncoder, indent=4)

def generate_report(actual,pre):
    result = {
        "Accuracy": accuracy_score(actual, pre),
        "F1_score": f1_score(actual, pre),
        "Precision": precision_score(actual, pre),
        "Recall": recall_score(actual, pre),
        "Confusion_Matrix": confusion_matrix(actual, pre)
    }

    return result

def model_evulation(x_train,y_train,x_test,y_test,model_dic):
    logging.info("Itrating on model dic")
    report={}
    for model_name,model in model_dic.items():
        # with mlflow.start_run(run_name=model_name):
            logging.info(f'fit the model {model_name}')
            model.fit(x_train,y_train)

            logging.info(f'Predict the model {model_name}')
            pre=model.predict(x_test)

            logging.info(f'Cross Validation of the model {model_name} on training data')
            train_score=cross_val_score(model,x_train,y_train,cv=5,scoring="accuracy",n_jobs=-1).mean()
            
            logging.info(f'Cross Validation of the model {model_name} on testing data')
            test_score=cross_val_score(model,x_test,y_test,cv=5,scoring="accuracy",n_jobs=-1).mean()
            
            logging.info(f'Final report of the model {model_name}')
            full_report=generate_report(actual=y_test,pre=pre)

            # # # Log parameters and metrics with MLflow
            # mlflow.log_param("model_name", model_name)
            # mlflow.log_metric("train_score", train_score)
            # mlflow.log_metric("test_score", test_score)
            # # mlflow.log_metrics(full_report)
            
            # # # Log the model
            # mlflow.sklearn.log_model(model, model_name)

            report[model_name]={
                "train_score":train_score,
                "test_score":test_score,
                "full_report":full_report
            }
    
    return report


def voting_classifier(x_train,y_train,x_test,y_test,model_dic,sorted_model):
    report={}
    estimator=[(model[0], model_dic[model[0]]) for model in sorted_model]
    # print(estimator)
    voting_clf = VotingClassifier(estimators=estimator,voting='soft')
    logging.info("voting successfull")
    voting_clf.fit(x_train,y_train)

    pre=voting_clf.predict(x_test)

    # logging.info(f'Cross Validation of the model {model_name} on training data')
    train_score=cross_val_score(voting_clf,x_train,y_train,cv=5,scoring="accuracy",n_jobs=-1).mean()
            
    # logging.info(f'Cross Validation of the model {model_name} on testing data')
    test_score=cross_val_score(voting_clf,x_test,y_test,cv=5,scoring="accuracy",n_jobs=-1).mean()
            
    # logging.info(f'Final report of the model {model_name}')
    full_report=generate_report(actual=y_test,pre=pre)

    report['voting_clf']={
                "train_score":train_score,
                "test_score":test_score,
                "full_report":full_report
            }

    return [voting_clf,report]


def find_top_models(report, top_n=3):
    sorted_models = sorted(report.items(), key=lambda item: item[1]["full_report"]["Recall"], reverse=True)
    return sorted_models[:top_n]
