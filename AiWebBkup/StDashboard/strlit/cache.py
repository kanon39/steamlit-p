"""
Streamlit cache functions
"""
import pandas as pd
import streamlit as st

from common import pre_process, age_fare


@st.cache_data
def load_titanic_data(csv):
    df1 = pd.read_csv(csv, encoding='utf8')
    df2 = pre_process(df1)
    df3 = age_fare(df2)
    return df1, df2, df3


@st.cache_data
def load_ai_pickle(pkl):
    import joblib
    dct = joblib.load(pkl)
    return dct
