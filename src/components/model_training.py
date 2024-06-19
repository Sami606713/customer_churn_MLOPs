from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.utils import model_evulation,save_report,save_file
import os
import logging
logging.basicConfig(level=logging.INFO)

def inisiate_model_training(train_array,test_array):
    counter=0
    logging.info("Model Training Start")
    # set the model and report path
    model_path=os.path.join("models","model.pkl")
    report_path=os.path.join(f"reports/{counter}","report.json")

    logging.info("Saperate Input and output columns")
    x_train=train_array[:,:-1]
    y_train=train_array[:,-1]
    x_test=test_array[:,:-1]
    y_test=test_array[:,-1]

    logging.info("Saperate Input and output columns")

    # make the models list
    models={
        "Lr":LogisticRegression(verbose=1,n_jobs=-1),
        "Dt":DecisionTreeClassifier(),
        "RF":RandomForestClassifier(verbose=1,n_jobs=-1),
        "xgboost":XGBClassifier(),  
        "knn":KNeighborsClassifier(n_jobs=-1)  
    }

    # Paramters Dict
    params={
        "Lr":{
            'penalty':['l2','l1'],
            'C':[0.1,0.01,0.001],
            'class_weight':["balanced",None],
            'solver':['newton-cholesky','liblinear']
        },
        "Dt":{
            "criterion":['gini','entropy','log_loss'],
            'max_depth':[10,20,30,50],
            'min_samples_split':[2,3,4],
            'min_samples_leaf':[1,2],
            'max_leaf_nodes':[2,None],
            'class_weight':['balanced',None,'balanced_subsample']
        },
        "RF":{
            'n_estimators':[100,200,300],
            'criterion':['gini','entropy','log_loss'],
            'max_depth':[10,20,30,50],
            'min_samples_split':[2,3,4],
            'min_samples_leaf':[1,2],
            'bootstrap':[True,False],
            'oob_score':[True,False],
            'class_weight':['balanced',None,'balanced_subsample'],
        },
        "xgboost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.2]
        },

        'knn':{
            'n_neighbors':[5,7,9,15],
            'weights':['uniform', 'distance']
        }

    }

    # Evulate the models
    final_report=model_evulation(x_train=x_train,y_train=y_train,
                                x_test=x_test,y_test=y_test,
                                model_dic=models,params=params) 
    print(final_report.keys())
    m=list(final_report.keys())[0]
    print(models[m])

    # Save the models report 
    save_report(file_path=report_path,report=final_report)

    # Save the model
    best_model=models[m]
    # set the model and report path
    model_path=os.path.join("models",f"{best_model[:-2]}.pkl")
    save_file(file_path= model_path,obj=best_model)


    