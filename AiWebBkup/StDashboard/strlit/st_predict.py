"""
Streamlit app for AI Predict
"""
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

from cache import load_titanic_data
from common import ai_predict

@st.cache_data
def load_ai_pickle(pkl):
    import joblib
    dct = joblib.load(pkl)
    return dct



begin = time.time()

BASE_DIR = Path.cwd()
CSV_FILE = BASE_DIR / 'DATA' / 'titanic_train.csv'
PICKLE_NAME = BASE_DIR / 'DATA' / 'dct0130.pkl'

# --- page
st.set_page_config(page_title='Predict', page_icon=':heart:')

# --- data prepare
dfTrain, dfPreN, dfCateN = load_titanic_data(CSV_FILE)
dctPickle = load_ai_pickle(PICKLE_NAME)

# --- sidebar
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
st.title("AI 예측")
st.write("Titanic 탑승자에 대한 생존 여부를 예측합니다.")
st.write("---")

st.write("##### 입력된 데이터")
widgetData = [pclass, sex, age, sibsp, parch, fare, cabin, embarked]
colNames = [c for c in dfCateN.columns if c != 'Survived']
dfWidget = pd.DataFrame(data=[widgetData], columns=colNames)
st.write(dfWidget)

predictData = [pclassDict[pclass], sex, age, sibsp, parch,
               fare, cabin, embarkedDict[embarked]]
dfPredict = pd.DataFrame(data=[predictData], columns=colNames)
"---"

pred = ai_predict(dctPickle, dfPredict)
if pred == 1:
    st.success("예측 결과 **:blue[생존]** 입니다.")
else:
    st.error("예측 결과 **:blue[사망]** 입니다.")
"---"

elapsed = time.time() - begin
st.info(f"{elapsed = :.1f} secs")
