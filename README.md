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
- In this step, we use the transformed data to train different models, after traing the model we can get the best model and then use for prediction.

# Use Case
- Clone the repo
```bash
git clone https://github.com/Sami606713/customer_churn_MLOPs
  ```

- Install Dependencies
```bash
pip install -r requirements.txt
  ```

- Second run the webapp.
```bash
streamlit run app.py
  ```
