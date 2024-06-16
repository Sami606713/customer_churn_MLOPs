from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.utils import model_evulation,save_report
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
        "Lr":LogisticRegression(verbose=1,class_weight='balanced',solver='liblinear',n_jobs=-1),
        "Dt":DecisionTreeClassifier(max_depth=10,              
                                    min_samples_split=10,     
                                    min_samples_leaf=4,       
                                    max_features='sqrt',      
                                    min_impurity_decrease=0.01,
                                    random_state=43,          
                                    class_weight='balanced'),
        "RF":RandomForestClassifier(),
        "xgboost":XGBClassifier(),  
        "knn":KNeighborsClassifier(n_neighbors=5,n_jobs=-1)  
    }

    # Evulate the models
    final_report=model_evulation(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,model_dic=models) 
    print(final_report)
    save_report(file_path=report_path,report=final_report)
    counter+=1

    