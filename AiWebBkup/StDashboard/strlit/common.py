"""
streamlit common functions
"""
import pandas as pd


# 불필요한 컬럼 삭제
def del_cols(df):
    df = df.copy()
    cols = ['PassengerId', 'Name', 'Ticket']
    df = df.drop(cols, axis=1)
    print(f"del_cols(): {df.shape=}")
    return df


# null 처리
def na_handle(df):
    df = df.copy()
    df['Age'] = df.Age.fillna(df.Age.mean()).astype('int')
    df['Cabin'] = df.Cabin.fillna('Z')
    df['Embarked'] = df.Embarked.fillna(df.Embarked.mode().values[0])
    df['Fare'] = df.Fare.fillna(df.Fare.mean())
    print(f"na_handle(): {df.shape=}")
    return df


# 그 외 처리
def pre_etc(df):
    df = df.copy()
    df['Cabin'] = df.Cabin.str[0]
    print(f"pre_etc(): {df.shape=}")
    return df


# 위 3개 모음
def pre_process(df):
    df = del_cols(df)
    df = na_handle(df)
    df = pre_etc(df)
    print(f"pre_process: {df.shape=}")
    return df


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


def ai_predict(pkl, df):
    def fn(col):
        if col.name in ['Sex', 'Cabin', 'Embarked', 'Age', 'Fare']:
            return pkl[col.name].transform(col)
        return col

    dfcate = age_fare(df)
    dflabel = dfcate.apply(fn, axis=0)
    scaled = pkl['scaler'].transform(dflabel)
    dfscale = pd.DataFrame(scaled, columns=df.columns)
    pred = pkl['model'].predict(dfscale)
    return pred