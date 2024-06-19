# Customer Churn Prediction 
- The goal of this project is to identify those customer who can leave and those who can't leave.

# Project Achieving
- Data Collection
- Data Transformation(like Scaliing,Encoding,or handling missing values)


# Data Collection
- First Step is to collect the data form `URL` and store them in a speccific location
[Data URL](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

- After collecting data next step i can do is to split the data into train and test set.

# Data Transformation
- After spliting the data i can get the train and test data so that i can prepare the data for model training pipeline.
- Building pipelines for data transforamtion.
    - `Numerical pipeline`
        - Scaling Numerical Data
    - `Categorical pipeline`
        - Encoding Categorical Data
- Building Transformer
    - After building pipelines combine all the pipeline using `ColumnsTransformers`

# Model Building
- In this step i can get the transform data and train o different models i-e.
```
models={
        "Lr":LogisticRegression(verbose=1,class_weight='balanced',solver='liblinear',n_jobs=-1),
        "Dt":DecisionTreeClassifier(max_depth=10,              
                                    min_samples_split=10,     
                                    min_samples_leaf=4,       
                                    max_features='sqrt',      
                                    min_impurity_decrease=0.01,
                                    random_state=43,          
                                    class_weight='balanced'),
        "RF":RandomForestClassifier(class_weight='balanced'),
        "xgboost":XGBClassifier(),  
        "knn":KNeighborsClassifier(n_neighbors=5,n_jobs=-1)  
    }
```
- Second i can check the result of all the model and based on `Recall` b/c in this project `Recall` is most important i can get the top3 model whose recall is high and make a big model using `VotingClassifier`
```
estimator=[(model[0], model_dic[model[0]]) for model in sorted_model]
    # print(estimator)
    voting_clf = VotingClassifier(estimators=estimator,voting='soft')
    logging.info("voting successfull")
    voting_clf.fit(x_train,y_train)
```
- At last i can save the model so the i can use further