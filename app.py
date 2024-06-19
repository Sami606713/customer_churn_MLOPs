import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import prediction

# Title of the app
st.title('Customer :blue[Churn] :rainbow: Prediction')

df=pd.read_csv('Data/raw/train.csv')
    
# set the input form
dic={}
# Row 1
col1,col2,col3,col4=st.columns(4)
with col1:
    gender=st.selectbox("gender",df['gender'].value_counts().index)
    dic['gender']=gender
with col2:
    SeniorCitizen=st.selectbox("SeniorCitizen",df['SeniorCitizen'].value_counts().index)
    dic['SeniorCitizen']=SeniorCitizen
with col3:
    Partner=st.selectbox("Partner",df['Partner'].value_counts().index)
    dic['Partner']=Partner
with col4:
    Dependents=st.selectbox("Dependents",df['Dependents'].value_counts().index)
    dic['Dependents']=Dependents

# Row 2
col1,col2,col3,col4=st.columns(4)
with col1:
    PhoneService=st.selectbox("PhoneService",df['PhoneService'].value_counts().index)
    dic['PhoneService']=PhoneService
with col2:
    MultipleLines=st.selectbox("MultipleLines",df['MultipleLines'].value_counts().index)
    dic['MultipleLines']=MultipleLines
with col3:
    InternetService=st.selectbox("InternetService",df['InternetService'].value_counts().index)
    dic['InternetService']=InternetService
with col4:
    OnlineSecurity=st.selectbox("OnlineSecurity",df['OnlineSecurity'].value_counts().index)
    dic['OnlineSecurity']=OnlineSecurity

# Row 3
col1,col2,col3,col4=st.columns(4)
with col1:
    OnlineBackup=st.selectbox("OnlineBackup",df['OnlineBackup'].value_counts().index)
    dic['OnlineBackup']=OnlineBackup
with col2:
    DeviceProtection=st.selectbox("DeviceProtection",df['DeviceProtection'].value_counts().index)
    dic['DeviceProtection']=DeviceProtection
with col3:
    TechSupport=st.selectbox("TechSupport",df['TechSupport'].value_counts().index)
    dic['TechSupport']=TechSupport
with col4:
    StreamingTV=st.selectbox("StreamingTV",df['StreamingTV'].value_counts().index)
    dic['StreamingTV']=StreamingTV

# Row 4
col1,col2,col3,col4=st.columns(4)
with col1:
    StreamingMovies=st.selectbox("StreamingMovies",df['StreamingMovies'].value_counts().index)
    dic['StreamingMovies']=StreamingMovies

with col2:
    Contract=st.selectbox("Contract",df['Contract'].value_counts().index)
    dic['Contract']=Contract
with col3:
    PaperlessBilling=st.selectbox("PaperlessBilling",df['PaperlessBilling'].value_counts().index)
    dic['PaperlessBilling']=PaperlessBilling
with col4:
    PaymentMethod=st.selectbox("PaymentMethod",df['PaymentMethod'].value_counts().index)
    dic['PaymentMethod']=PaymentMethod

# Row 5
col1,col2,col3=st.columns(3)
with col1:
    tenure=st.number_input(f'tenure',min_value=0)
    dic['tenure']=tenure
with col2:
    MonthlyCharges=st.number_input(f'MonthlyCharges',min_value=0)
    dic['MonthlyCharges']=MonthlyCharges
with col3:
    TotalCharges=st.number_input(f'TotalCharges',min_value=0)
    dic['TotalCharges']=TotalCharges

if st.button("Predict"):
    df=pd.DataFrame(dic,index=[0])
    st.dataframe(df)
    # st.success("prediction")
    result=prediction(df)
    if result[0]==0:
        # st.success(result[0])
        st.success("Customer can't leave the platform")
    elif(result[0]==1):
        st.error("Customer can leave the platform")
    