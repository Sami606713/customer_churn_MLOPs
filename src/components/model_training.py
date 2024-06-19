from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
<<<<<<< HEAD
from src.utils import model_evulation,save_report,save_file
=======
from src.utils import model_evulation,save_report,voting_classifier,find_top_models,save_file
>>>>>>> d9fce202e651b27630c772bc1c032d56e5014d69
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
<<<<<<< HEAD
        "Lr":LogisticRegression(verbose=1,n_jobs=-1),
        "Dt":DecisionTreeClassifier(),
        "RF":RandomForestClassifier(verbose=1,n_jobs=-1),
=======
        "Lr":LogisticRegression(verbose=1,class_weight='balanced',solver='liblinear',n_jobs=-1),
        "Dt":DecisionTreeClassifier(max_depth=10,              
                                    min_samples_split=10,     
                                    min_samples_leaf=4,       
                                    max_features='sqrt',      
                                    min_impurity_decrease=0.01,
                                    random_state=43,          
                                    class_weight='balanced'),
        "RF":RandomForestClassifier(class_weight='balanced'),
>>>>>>> d9fce202e651b27630c772bc1c032d56e5014d69
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
<<<<<<< HEAD
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

=======
    final_report:dict=model_evulation(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,model_dic=models) 
    # print(final_report)
    save_report(file_path=report_path,report=final_report)
    # counter+=1
    sorted_model=find_top_models(final_report)
    
    logging.info("Call the voting classifier")
    final_model,full_report_2=voting_classifier(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,model_dic=models,sorted_model=sorted_model)

    logging.info(f"final model at this location {model_path}")
    save_file(model_path,final_model)

    print(full_report_2)
>>>>>>> d9fce202e651b27630c772bc1c032d56e5014d69

    