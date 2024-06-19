# Customer Churn Prediction 
The goal of this project is to identify customers who are likely to leave (churn) and those who are not.

## Project Achievements
- Data Collection
- Data Transformation (like Scaling, Encoding, or Handling Missing Values)

## Data Collection
- The first step is to collect the data from the provided URL and store it in a specific location.
  [Data URL](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- After collecting the data, the next step is to split the data into training and test sets.

## Data Transformation
- After splitting the data, we prepare the train and test datasets for the model training pipeline.
- Building pipelines for data transformation:
    - **Numerical Pipeline**
        - Scaling Numerical Data
    - **Categorical Pipeline**
        - Encoding Categorical Data
- Building the Transformer:
    - After building the pipelines, combine them using `ColumnTransformer`.

## Model Building
- In this step, we use the transformed data to train different models, such as:
```python
models = {
    "Lr": LogisticRegression(verbose=1, n_jobs=-1),
    "Dt": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(verbose=1, n_jobs=-1),
    "xgboost": XGBClassifier(),  
    "knn": KNeighborsClassifier(n_jobs=-1)  
}
```
- Models Paramters
```python
params = {
    "Lr": {
        'penalty': ['l2', 'l1'],
        'C': [0.1, 0.01, 0.001],
        'class_weight': ["balanced", None],
        'solver': ['newton-cholesky', 'liblinear']
    },
    "Dt": {
        "criterion": ['gini', 'entropy', 'log_loss'],
        'max_depth': [10, 20, 30, 50],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2],
        'max_leaf_nodes': [2, None],
        'class_weight': ['balanced', None, 'balanced_subsample']
    },
    "RF": {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [10, 20, 30, 50],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False],
        'oob_score': [True, False],
        'class_weight': ['balanced', None, 'balanced_subsample']
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2]
    },
    "knn": {
        'n_neighbors': [5, 7, 9, 15],
        'weights': ['uniform', 'distance']
    }
}
```

# Use Case
- First install `requirement.txt` file.
python```
pip install -r requirements.txt
```

- Second run the webapp.
```python
streamlit run app.py
```
