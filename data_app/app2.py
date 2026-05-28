import streamlit as st
import pandas as pd

from tabs.distroTab import show_dist
from tabs.dataTab import show_data
from tabs.relaTab import show_rela
from tabs.cleaningTab import show_cleaning

st.title("A Simple CSV Visualizer")

file = st.file_uploader("Upload a CSV file", type="csv")

def generateDataReport(data):
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()

    dataReport = {
        "Numerical" : numerical_cols + datetime_cols,
        "Categorical" : categorical_cols,
        "Distribution" : numerical_cols
    }

    return dataReport

if file:
    data = pd.read_csv(file).dropna()
    data = data.convert_dtypes()
    if data.empty:
        st.warning("The dataset is empty after removing rows with missing values. Please upload a CSV with more complete data or adjust missing value handling.")
    else:
        dataReport = generateDataReport(data)
        numeric = dataReport["Numerical"]
        categorical = dataReport["Categorical"]
        distro = dataReport["Distribution"]

        print(dataReport["Numerical"])
        print(dataReport["Categorical"])
        print(dataReport["Distribution"])

        ## --- main app tabs ---
        dataTab, cleanTab, distTab, relTab= st.tabs(["Data", "Cleaning", "Distributions", "Relationships"])

        with dataTab:
            show_data(data)

        with cleanTab:
            show_cleaning(numeric, data)

        with distTab:
            show_dist(distro, categorical, data)

        with relTab:
            show_rela(numeric, categorical, data)
