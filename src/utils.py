import pandas as pd
import pickle as pkl
import yaml
# import mlflow
# import mlflow.sklearn
import json
import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,classification_report
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

def save_params(parm_file,path):
    with open(path, 'w') as yaml_file:
        yaml.dump(parm_file, yaml_file)

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

def model_evulation(x_train,y_train,x_test,y_test,model_dic,params):
    logging.info("Itrating on model dic")
    report={}
    best_params={}
    for model_name,model in model_dic.items():
        param_grid=params.get(model_name,{})
        
        # grid_search=RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
        #                                 scoring='accuracy', cv=5, verbose=1, n_jobs=-1,n_iter=50)
        # grid_search=GridSearchCV(estimator=model, param_grid=param_grid, 
                                        # scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

        model.fit(x_train,y_train)
        # grid_search.fit(x_train,y_train)

        # best_model=grid_search.best_estimator_

        y_pred=model.predict(x_test)
        # y_pred=grid_search.predict(x_test)

        logging.info(f'Cross Validation of the model {model_name} on training data')
        train_score=cross_val_score(model,x_train,y_train,cv=5,scoring="accuracy",n_jobs=-1).mean()
            
        logging.info(f'Cross Validation of the model {model_name} on testing data')
        test_score=cross_val_score(model,x_test,y_test,cv=5,scoring="accuracy",n_jobs=-1).mean()
            
        logging.info(f'Final report of the model {model_name}')
        full_report=generate_report(actual=y_test,pre=y_pred)

        # Store the best parameters
        # best_params[model_name] = grid_search.best_params_
        

        report[model_name]={
                'model':model
                "train_score":train_score,
                "test_score":test_score,
                "full_report":full_report,
                # "class_report":class_report
        }

    # Save the parameters
    save_params(parm_file=best_params,path=os.path.join("models",f"best_params.yml"))

    # Sorting the report dictionary based on test_score
    sorted_report = dict(sorted(report.items(), key=lambda item: item[1]['test_score'], reverse=True))
    
    return sorted_report



