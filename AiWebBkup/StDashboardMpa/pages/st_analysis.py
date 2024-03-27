"""
Streamlit app for Interactive Data Analysis
"""
import time

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from utils.common import pre_process


@st.cache_data
def load_titanic_data():
    df1 = pd.read_csv('./DATA/titanic_train.csv', encoding='utf8')
    df2 = pre_process(df1)
    return df1, df2


begin = time.time()

# --- page
st.set_page_config(page_title='Analysis', page_icon=':smile:')

# --- data prepare
dfTrain, dfPreN = load_titanic_data()

# --- sidebar
colOptions = list(dfPreN.columns)
col = st.sidebar.selectbox("컬럼을 선택하세요.", options=colOptions)
st.sidebar.write("---")

hueOptions = [None] + colOptions
hue = st.sidebar.selectbox("hue 를 선택하세요.", options=hueOptions)
st.sidebar.write("---")

bins = st.sidebar.slider("bins 를 선택하세요.", 5, 30, 10, 5)

# --- body
st.title("Titanic Data Analysis (interacive)")

tabs = st.tabs(['dataframe', 'countplot', 'barplot', 'histplot', 'scatterplot'])
with tabs[0]:
    st.write(f"{dfPreN.shape = }")
    st.dataframe(dfPreN)
    st.write(pd.concat([dfPreN.dtypes, dfPreN.count()], axis=1).T)

with tabs[1]:
    fig = plt.figure()
    sns.countplot(data=dfPreN, x=col, hue=hue)
    st.pyplot(fig)

with tabs[2]:
    st.write("##### 컬럼별 생존율")
    fig = plt.figure()
    sns.barplot(data=dfPreN, x=col, y='Survived', hue=hue)
    st.pyplot(fig)

with tabs[3]:
    fig = plt.figure()
    sns.histplot(data=dfPreN, x=col, hue=hue, bins=bins)
    st.pyplot(fig)

with tabs[4]:
    st.write("나이와 요금간 상관관계")
    fig = plt.figure()
    sns.scatterplot(data=dfPreN, x='Age', y='Fare')
    st.pyplot(fig)

elapsed = time.time() - begin
st.info(f"{elapsed = :.1f} secs")


