"""
Streamlit app for AI Predict
"""
import time
import joblib
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# from utils.common import pre_process
from AiWebBkup.StDashboard.strlit.cache import load_titanic_data, load_ai_pickle
from AiWebBkup.StDashboard.strlit.common import ai_predict

# @st.cache_data
# def load_titanic_data(csv):
#     df1 = pd.read_csv(csv, encoding='utf8')
#     df2 = pre_process(df1)
#     df3 = age_fare(df2)
#     return df1, df2, df3


begin = time.time()

BASE_DIR = Path.cwd()
CSV_FILE = BASE_DIR / 'DATA' / 'titanic_train.csv'
PICKLE_NAME = BASE_DIR / 'DATA' / 'dct0130.pkl'

# --- page
st.set_page_config(page_title='Predict', page_icon=':heart:')

# --- data prepare
dfTrain, dfPreN, dfCateN = load_titanic_data(CSV_FILE)
# dctPickle = joblib.load(PICKLE_NAME)
dctPickle = load_ai_pickle(PICKLE_NAME)

# --- sidebar
# pclassOpts = dfCateN.Pclass.unique()
# sexOpts = dfCateN.Sex.unique()
# ageOpts = dfCateN.Age.unique()
# sibspOpts = dfCateN.SibSp.unique()
# parchOpts = dfCateN.Parch.unique()
# fareOpts = dfCateN.Fare.unique()
# cabinOpts = dfCateN.Cabin.unique()
# embarkedOpts = dfCateN.Embarked.unique()
#
# pclass = st.sidebar.selectbox("Pclass 선택", options=pclassOpts)
# sex = st.sidebar.selectbox("Sex 선택", options=sexOpts)
# age = st.sidebar.selectbox("Age 선택", options=ageOpts)
# sibsp = st.sidebar.selectbox("SibSp 선택", options=sibspOpts)
# parch = st.sidebar.selectbox("Parch 선택", options=parchOpts)
# fare = st.sidebar.selectbox("Fare 선택", options=fareOpts)
# cabin = st.sidebar.selectbox("Cabin 선택", options=cabinOpts)
# embarked = st.sidebar.selectbox("Embarked 선택", options=embarkedOpts)

pclassDict = {'First': 1, 'Second': 2, 'Third': 3}
sexOpts = dfCateN.Sex.unique()
ageMin = dfPreN.Age.min()
ageMax = dfPreN.Age.max()
sibspMin = dfCateN.SibSp.min()
sibspMax = dfCateN.SibSp.max()
parchMin = dfCateN.Parch.min()
parchMax = dfCateN.Parch.max()
fareMin = dfPreN.Fare.min()
fareMax = dfPreN.Fare.max()
cabinOpts = sorted(dfCateN.Cabin.unique())
embarkedDict = {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}

pclass = st.sidebar.selectbox("Pclass 선택", options=pclassDict.keys())
sex = st.sidebar.selectbox("Sex 선택", options=sexOpts)
age = st.sidebar.number_input("Age 선택", ageMin, ageMax, 5, 2)
sibsp = st.sidebar.slider("SibSp 선택", sibspMin, sibspMax, 0, 1)
parch = st.sidebar.slider("Parch 선택", parchMin, parchMax, 0, 1)
fare = st.sidebar.number_input("Fare 선택", fareMin, fareMax, 100.0, 10.0)
cabin = st.sidebar.selectbox("Cabin 선택", options=cabinOpts)
embarked = st.sidebar.selectbox("Embarked 선택", options=embarkedDict.keys())

# --- body
st.title("Streamlit AI 예측")
st.write("Titanic 탑승자에 대한 생존 여부를 예측합니다.")
# st.write(dfCateN.head())
st.write("---")

st.write("##### 입력된 데이터")
# # dfOne = dfCateN.head(1).drop('Survived', axis=1)
# dfOne = dfCateN.iloc[1:2].drop('Survived', axis=1)

widgetData = [pclass, sex, age, sibsp, parch, fare, cabin, embarked]
colNames = [c for c in dfCateN.columns if c != 'Survived']
dfWidget = pd.DataFrame(data=[widgetData], columns=colNames)
st.write(dfWidget)

predictData = [pclassDict[pclass], sex, age, sibsp, parch,
               fare, cabin, embarkedDict[embarked]]
dfPredict = pd.DataFrame(data=[predictData], columns=colNames)
"---"


# def fn(col):
#     if col.name in ['Sex', 'Cabin', 'Embarked', 'Age', 'Fare']:
#         return dctPickle[col.name].transform(col)
#     return col
#
# dfZlabel = dfZero.apply(fn, axis=0)
# scaled = dctPickle['scaler'].transform(dfZlabel)
# dfZscale = pd.DataFrame(scaled, columns=colNames)
# dctPickle['model'].predict(dfZscale)

# pred = ai_predict(dctPickle, dfWidget)
pred = ai_predict(dctPickle, dfPredict)
if pred == 1:
    st.success("예측 결과 **:blue[생존]** 입니다.")
else:
    st.error("예측 결과 **:blue[사망]** 입니다.")
"---"


elapsed = time.time() - begin
st.info(f"{elapsed = :.1f} secs")


