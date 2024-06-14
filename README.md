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