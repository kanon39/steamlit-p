#!/usr/bin/env python
# coding: utf-8

# # Analysis and Predict for Titanic data (dfCateN)

# ### 처리 순서
# ##### 1. Train/Test 데이터 준비
# ##### 2. 데이터 기본정보 확인
# ##### 3. 데이터 전처리
# ######  &emsp; 3-1. 불필요한 컬럼 삭제
# ######  &emsp; 3-2. NaN 데이터 처리
# ######  &emsp; 3-3. 그외 전처리
# ##### 4. Encoding
# ##### 5. Scaling
# ##### 6. train_test_split()
# ##### 7. 모델 학습/평가
# ##### 8. 실제 예측

# In[ ]:





# ##### import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### 1. Train/Test 데이터 준비

# In[2]:


from pathlib import Path
Path.cwd()


# In[3]:


TRAIN_DATA = Path.cwd().parent / 'DATA/titanic_train.csv'
TEST_DATA = Path.cwd().parent / 'DATA/titanic_test.csv'


# In[4]:


dfTrain = pd.read_csv(TRAIN_DATA)
dfTrain.shape


# In[5]:


dfTest = pd.read_csv(TEST_DATA)
dfTest.shape


# ### 2. 데이터 기본 정보 확인

# ##### 데이터의 내용을 확인한다.

# In[6]:


dfTrain.head()


# In[7]:


dfTest.head()


# ##### Null 데이터 확인한다.

# ##### np.nan 은 float type, 만일 있으면 int 컬럼이 --> float 컬럼이 된다.

# In[8]:


dfTrain.info()


# In[9]:


dfTest.info()


# ##### describe() 는 number type 만 보여줌.

# In[10]:


dfTrain.describe()


# In[11]:


dfTest.describe()


# ### 3. 데이터 전처리

# #####  AI 알고리즘을 실행하기 전에, 데이터를 목적에 맞게 변환한다.
# ##### EDA (Exploratory Data Analysis), 탐색적 데이터 분석을 통해, 데이터 특성을 파악한다.
# ##### --------------------------------------------------------------------------------------------
# ##### EDA (Exploratory Data Analysis), 탐색적 데이터 분석
# ###### &emsp; - 데이터를 다양한 각도에서 관찰하고 이해하는 과정
# ###### &emsp; - 표, 그래프, 통계값 등 다양한 지식 등을 활용, (예를 들어) 이상치, 속성간의 관계, 주요한 속성 등을 찾아낸다.
# ###### &emsp; *. 이 과정을 확장하여, 데이터 분석을 통해 [데이터 인사이트]를 발굴할 수도 있다.

# In[ ]:





# ##### EDA (매우 다양하게 진행할 수 있다)

# In[12]:


dfTrain.Survived.value_counts()


# In[13]:


sns.countplot(data=dfTrain, x='Survived', hue='Sex')


# ### 3-1. 불필요한 컬럼 삭제

# ##### -. target 컬럼에 영향을 주는 feature(컬럼) 를 선택 --> feature selection 은 중요한 분야임
# ##### -. 여기서는 직관적으로 불필요한 컬럼을 선택함

# In[ ]:





# ##### 함수로 만들면, train 및 test 데이터에 동일하게 적용할 수 있다.

# In[14]:


# 불필요한 컬럼 삭제
def del_cols(df):
    df = df.copy()
    cols = ['PassengerId', 'Name', 'Ticket']
    df = df.drop(cols, axis=1)
    print(f"del_cols(): {df.shape=}")
    return df


# In[15]:


dfDrop = del_cols(dfTrain)
dfDrop.shape


# ### 3-2. NaN 처리

# ##### -. NaN 값이 많은 컬럼은 삭제한다.
# ##### -. 아니면, NaN 을 다른 값으로 바꾼다. (평균값, 중간값, 최빈값 등)

# In[16]:


cols = ['Age', 'Fare', 'Cabin', 'Embarked']
for c in cols:
    print('-' * 30, c)
    print(dfDrop[c].value_counts())


# In[17]:


dfDrop.Embarked.mode()


# In[18]:


dfDrop.Age.mean()


# In[19]:


dfDrop.Fare.mean()


# ##### Cabin 컬럼은 나중에 첫 글자만 추출할 것이므로, NaN -> Z 로 치환한다. (현 데이터에 Z 사용 안하므로)

# In[ ]:





# In[20]:


# null 처리
def na_handle(df):
    df = df.copy()
    df['Age'] = df.Age.fillna(df.Age.mean()).astype('int')
    df['Cabin'] = df.Cabin.fillna('Z')
    df['Embarked'] = df.Embarked.fillna(df.Embarked.mode().values[0])
    df['Fare'] = df.Fare.fillna(df.Fare.mean())    
    print(f"na_handle(): {df.shape=}")
    return df


# In[21]:


dfNa = na_handle(dfDrop)
dfNa.shape


# ### 3-3. 그 외 처리

# ##### -. Cabin 컬럼의 첫 글자는 의미가 있다.

# In[22]:


# 그 외 처리
def pre_etc(df):
    df = df.copy()
    df['Cabin'] = df.Cabin.str[0]
    print(f"pre_etc(): {df.shape=}")
    return df


# In[23]:


dfEtc = pre_etc(dfNa)
dfEtc.shape


# In[ ]:





# ### 전처리 함수들 정리

# In[24]:


# 위 3개 모음
def pre_process(df):
    df = del_cols(df)
    df = na_handle(df)
    df = pre_etc(df)
    print(f"pre_process: {df.shape=}")
    return df


# In[25]:


dfPreN = pre_process(dfTrain)


# In[26]:


dfPreT = pre_process(dfTest)


# In[ ]:





# ##### 전처리 후, 데이터 확인

# In[27]:


dfPreN.Survived.value_counts()


# In[28]:


dfPreN.isna().sum().sum(), dfPreT.isna().sum().sum()


# ### 3-4. Age/Fare Category

# In[29]:


def age_fare(df):
    def convert_age(v):
        if v <= 5:
            return 'Child'
        elif v < 18:
            return 'Student'
        elif v < 48:
            return 'Youth'
        elif v < 65:
            return 'Adult'
        else:
            return 'Senior'

    def convert_fare(v):
        if v < 50:
            return '<50'
        elif v < 150:
            return '<150'
        else:
            return '>=150'

    df = df.copy()
    df['Age'] = df.Age.apply(convert_age)
    df['Fare'] = df.Fare.apply(convert_fare)
    return df


# In[30]:


dfCateN = age_fare(dfPreN)
dfCateT = age_fare(dfPreT)
dfCateN.shape, dfCateT.shape


# In[31]:


dfCateN.head()


# In[ ]:





# ### 4. Label Encoding

# ##### (1) 문자열 컬럼에, 각 컬럼별로 수행한다. (2) Train 과 Test 에 동일한 Encoder 를 사용한다.
# ##### (참고) One-Hot Encoding

# In[32]:


from sklearn.preprocessing import LabelEncoder


# In[33]:


def label_encode(df1, df2):
    le = LabelEncoder()
    
    df1 = df1.copy()
    df2 = df2.copy()
    for c in df1.select_dtypes('object').columns:
        le.fit(df1[c])
        df1[c] = le.transform(df1[c])
        df2[c] = le.transform(df2[c])
    print(f"label_encode(): {df1.shape=}, {df2.shape=}")
    return df1, df2


# In[34]:


dfLabelN, dfLabelT = label_encode(dfCateN, dfCateT)


# In[ ]:





# ### 5. Standard Scaling

# ##### (1) 모든 컬럼에, 한꺼번에 수행한다. (2) Train 과 Test 에 동일한 Scaler 를 사용한다.
# ##### (참고) MinMaxScaling

# In[35]:


from sklearn.preprocessing import StandardScaler


# In[36]:


def standard_scale(df1, df2):
    scaler = StandardScaler()
    
    df1 = df1.copy()
    df2 = df2.copy()
    target = df1.pop('Survived')
    
    scaler.fit(df1)
    scale1 = scaler.transform(df1)
    scale2 = scaler.transform(df2)
    
    tmp = pd.DataFrame(scale1, columns=df1.columns)
    df1 = pd.concat([target, tmp], axis=1)
    df2 = pd.DataFrame(scale2, columns=df2.columns)
    
    print(f"standard_scale(): {df1.shape=}, {df2.shape=}")
    return df1, df2


# In[37]:


dfScaleN, dfScaleT = standard_scale(dfLabelN, dfLabelT)


# In[ ]:





# ### 6. train_test_split()

# ##### dfTrain 대상, dfTest 는 적용대상 아님
# ##### &emsp; -. Train 데이터는 모델 학습용, Test 데이터는 모델 평가용
# ##### &emsp; -. Valid 데이터는 과적합(Overfitting) 방지용
# ##### stratify=y : y 데이터의 비율을 유지하면서, 데이터를 분리함

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X = dfScaleN.drop('Survived', axis=1)
y = dfScaleN.Survived

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=1004, stratify=y)
print(f"{trainX.shape=}, {testX.shape=}, {trainy.shape=}, {testy.shape=}")


# In[ ]:





# # 7. 모델 학습/평가

# ##### 처리 과정 : 알고리즘 선택(초기 모델) -> 학습 -> 평가 --(반복)--> 최종 모델
# ##### 분류 알고리즘 예시 : 결정 트리, 랜덤 포레스트, 로지스틱 회귀

# ### Decision Tree, 결정 트리 알고리즘

# In[40]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[43]:


# Decision Tree 정확도 : 0.7654
# dt = DecisionTreeClassifier(random_state=1004)
# dt.fit(trainX, trainy)
# dtPreds = dt.predict(testX)

# acc = accuracy_score(testy, dtPreds)
# print(f"Decision Tree 정확도 : {acc:.4f}")


# In[42]:


dt = DecisionTreeClassifier(random_state=1004)
dt.fit(trainX, trainy)
dtPreds = dt.predict(testX)

acc = accuracy_score(testy, dtPreds)
print(f"Decision Tree 정확도 : {acc:.4f}")


# ### 다른 알고리즘

# In[44]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[40]:


# Random Forest 정확도 : 0.7989
# Logistic Regression 정확도 : 0.8045

# rf = RandomForestClassifier(random_state=1004)
# lr = LogisticRegression(random_state=1004)

# rf.fit(trainX, trainy)
# lr.fit(trainX, trainy)

# rfPreds = rf.predict(testX)
# lrPreds = lr.predict(testX)

# rfAcc = accuracy_score(testy, rfPreds)
# lrAcc = accuracy_score(testy, lrPreds)
# print(f"Random Forest 정확도 : {rfAcc:.4f}")
# print(f"Logistic Regression 정확도 : {lrAcc:.4f}")


# In[45]:


rf = RandomForestClassifier(random_state=1004)
lr = LogisticRegression(random_state=1004)

rf.fit(trainX, trainy)
lr.fit(trainX, trainy)

rfPreds = rf.predict(testX)
lrPreds = lr.predict(testX)

rfAcc = accuracy_score(testy, rfPreds)
lrAcc = accuracy_score(testy, lrPreds)
print(f"Random Forest 정확도 : {rfAcc:.4f}")
print(f"Logistic Regression 정확도 : {lrAcc:.4f}")


# In[ ]:





# ### 8. 실제 예측

# ##### 최종 모델을 만든 후에, 실제 데이터를 예측한다.
# ##### target 컬럼이 없는 데이터에, 최종 모델을 적용해 값을 예측함.
# ##### (최종 모델 활용 예시) : 실제 데이터 예측, 해커톤 답안 제출, 웹 프로그램에 활용, 등등등

# In[46]:


# dfTest = pd.read_csv(TEST_DATA)
# dfScaleT = standard_scale(dfLabelT)

# REAL_DATA, dfReal, dfPreR, dfLabelR, dfScaleR
dfScaleR = dfScaleT
dfScaleR.shape


# In[47]:


preds = lr.predict(dfScaleR)
print(f"{preds.shape=}")


# In[48]:


preds[:10]


# In[ ]:





# # AI Predict 예시

# ##### 개별 데이터 예측 - DataFrame or 2d array 입력

# In[49]:


dfScaleR.head()


# In[50]:


row = '0.827377	0.737695	0.473014	-0.474545	-0.473674	0.282598	0.522067	-0.678175'
arr = [float(i) for i in row.split()]
arr


# In[51]:


lr.predict([arr])


# ##### 데이터 변환 : dfTrain -> dfPreN -> dfCateN -> dfLabelN -> dfScaleN
# ##### dfCateN 데이터 사용하고, label/scale 처리 필요함

# In[52]:


dfCateN.head()


# ##### pickle and save

# In[53]:


dctPickle = {}


# In[54]:


from copy import copy

def label_encode_pickle(df1, df2):
    le = LabelEncoder()
    
    df1 = df1.copy()
    df2 = df2.copy()
    for c in df1.select_dtypes('object').columns:
        le.fit(df1[c])
        df1[c] = le.transform(df1[c])
        df2[c] = le.transform(df2[c])
        dctPickle[c] = copy(le)
    print(f"label_encode_pickle(): {df1.shape=}, {df2.shape=}")
    return df1, df2


# In[55]:


dfLabelN, dfLabelT = label_encode_pickle(dfCateN, dfCateT)


# In[56]:


def standard_scale_pickle(df1, df2):
    scaler = StandardScaler()
    
    df1 = df1.copy()
    df2 = df2.copy()
    target = df1.pop('Survived')
    
    scaler.fit(df1)
    scale1 = scaler.transform(df1)
    scale2 = scaler.transform(df2)
    dctPickle['scaler'] = scaler
    
    tmp = pd.DataFrame(scale1, columns=df1.columns)
    df1 = pd.concat([target, tmp], axis=1)
    df2 = pd.DataFrame(scale2, columns=df2.columns)
    
    print(f"standard_scale_pickle(): {df1.shape=}, {df2.shape=}")
    return df1, df2


# In[57]:


dfScaleN, dfScaleT = standard_scale_pickle(dfLabelN, dfLabelT)


# In[58]:


dctPickle['model'] = lr


# In[59]:


dctPickle


# In[60]:


dctPickle['Sex'].classes_


# In[61]:


dctPickle['Cabin'].classes_


# In[62]:


dctPickle['Embarked'].classes_


# ##### AI 객체들 저장

# In[63]:


import joblib


# In[64]:


Path.cwd()


# In[65]:


get_ipython().system(' dir ..\\DATA')


# In[66]:


PICKLE_NAME = Path.cwd().parent / 'DATA/dct0130.pkl'

joblib.dump(dctPickle, PICKLE_NAME)


# In[67]:


get_ipython().system(' dir ..\\DATA')


# In[ ]:





# ### AI Predict 예측 Web 모의 실행

# In[68]:


dfCateN.head()


# ##### AI 객체들 불러오기

# In[69]:


dctPickle = joblib.load(PICKLE_NAME)
dctPickle


# ##### 예측 결과 = 0 (사망)

# In[70]:


row = '3	male	Youth	1	0	<50	Z	S'
lst = row.split()
lst


# In[71]:


colNames = [c for c in dfPreN.columns if c != 'Survived']
dfZero = pd.DataFrame([lst], columns=colNames)
dfZero


# In[72]:


# dfZero['Sex'] = dctPickle['Sex'].transform(dfZero['Sex'])

def fn(col):
    if col.name in ['Sex', 'Cabin', 'Embarked', 'Age', 'Fare']:
        return dctPickle[col.name].transform(col)
    return col

dfZlabel = dfZero.apply(fn, axis=0)
dfZlabel


# In[73]:


scaled = dctPickle['scaler'].transform(dfZlabel)
dfZscale = pd.DataFrame(scaled, columns=colNames)
dfZscale


# In[74]:


dctPickle['model'].predict(dfZscale)


# ##### 예측 결과 = 1 (생존)

# In[75]:


row = '1	female	Youth	1	0	<150	C	C'
lst = row.split()
lst


# In[76]:


dfOne = pd.DataFrame([lst], columns=colNames)
dfOlabel = dfOne.apply(fn, axis=0)
scaled = dctPickle['scaler'].transform(dfOlabel)
dfOscale = pd.DataFrame(scaled, columns=colNames)
dfOscale


# In[77]:


dctPickle['model'].predict(dfOscale)


# In[ ]:




