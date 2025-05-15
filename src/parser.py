import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from imblearn.over_sampling import SMOTE

def feature_engineering(df):
    median_emp_length = df['person_emp_length'].replace(0, np.nan).median()
    df['person_emp_length'] = df['person_emp_length'].replace(0, median_emp_length)

    df['financial_burden'] = df['loan_amnt'] * df['loan_int_rate']
    df['income_per_year_emp'] = df['person_income'] / (df['person_emp_length'] + 1e-5)
    df['int_per_year_emp'] = df['loan_int_rate'] / (df['person_emp_length'] + 1e-5)
    return df


def preprocess_train(file_path: str, balance_classes: bool = False, use_smote: bool = False, use_feature_eng: bool = False):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['id'])

    if use_feature_eng:
        df = feature_engineering(df)

    # One-hot encoding 
    df = pd.get_dummies(df, columns=[
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ], drop_first=True)

    df = df.dropna()
    X = df.drop(columns=['loan_status'])  
    y = df['loan_status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Divide em treino/teste 
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    if balance_classes:
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            class_weights_dict = None
        else:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights_dict = dict(zip(np.unique(y_train), class_weights))
        return X_train, X_val, y_train, y_val, scaler, X.columns, class_weights_dict

    return X_train, X_val, y_train, y_val, scaler, X.columns


def preprocess_test(file_path: str, scaler, columns, use_feature_eng: bool = False):
    df = pd.read_csv(file_path)
    ids = df['id']
    df = df.drop(columns=['id'])

    if use_feature_eng:
        df = feature_engineering(df)

    df = pd.get_dummies(df, columns=[
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ], drop_first=True)

    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]  # Reordenar colunas

    X_scaled = scaler.transform(df)

    return X_scaled, ids
