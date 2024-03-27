"""
streamlit dashboard, main file of MPA
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from utils.cache import load_titanic_data

begin = time.time()

BASE_DIR = Path.cwd()
CSV_FILE = BASE_DIR / 'DATA' / 'titanic_train.csv'

# --- page
st.set_page_config(page_title='Dashboard', page_icon='✨', layout='wide')

# --- data prepare
dfTrain, dfPreN, dfCateN = load_titanic_data(CSV_FILE)
# --- sidebar

# --- body
st.title("Streamlit Dashboard for Titanic Data")
"---"

st.markdown("##### 알고리즘 정확도")

style = """
    <style>
        div[data-testid='stMetric'] {
            border: 2px solid #ccc;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
"""
st.markdown(style, unsafe_allow_html=True)

cols = st.columns([1, 1, 1, 1])
with cols[0]:
    st.metric("Decision Tree", 78.2, -1.3)

with cols[1]:
    st.metric("Random Forest", 79.3, -0.2)

with cols[2]:
    st.metric("Logistic Regrssion", 81.0, 1.5)

with cols[3]:
    st.metric("Average Accuracy", 79.5, 0)
"---"

st.markdown("##### 주요 컬럼의 생존율")
cols = st.columns([1, 1, 1])
with cols[0]:
    st.markdown("**Pclass**")
    fig = plt.figure()
    sns.countplot(data=dfCateN, x='Pclass', hue='Survived')
    st.pyplot(fig)

with cols[1]:
    st.markdown("**Sex**")
    fig = plt.figure()
    sns.countplot(data=dfCateN, x='Sex', hue='Survived')
    st.pyplot(fig)

with cols[2]:
    st.markdown("**Age**")
    order = ['Child', 'Student', 'Youth', 'Adult', 'Senior']
    fig = plt.figure()
    sns.countplot(data=dfCateN, x='Age', hue='Survived', order=order)
    st.pyplot(fig)
"---"

st.markdown("##### 데이터 인사이트")
with st.expander("데이터 분석을 통해 Insight 를 도출한다.", expanded=True):
    html = """
        <ul>
            <li>생존율에 영향을 미치는 컬럼은, Pclass, Sex, Age 이다.</li>
            <li>1등급 승객이 생존 가능성이 높다.</li>
            <li>남성보다는 여성이 생존율이 높다.</li>
            <li>나이가 10세 미만인 경우 생존율이 높다.</li>
        </ul>
    """
    st.markdown(html, unsafe_allow_html=True)
"---"


elapsed = time.time() - begin
st.info(f"Elapsed : {elapsed:.1f} secs")
