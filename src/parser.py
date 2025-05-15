import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def feature_engineering(df):
    median_emp_length = df['person_emp_length'].replace(0, np.nan).median()
    df['person_emp_length'] = df['person_emp_length'].replace(0, median_emp_length)

    df['financial_burden'] = df['loan_amnt'] * df['loan_int_rate'] / 100  

    df['income_per_year_emp'] = df['person_income'] / (df['person_emp_length'] + 1e-5)
    df['int_per_year_emp'] = df['loan_int_rate'] / (df['person_emp_length'] + 1e-5)

    df['loan_to_credit_hist'] = df['loan_amnt'] / (df['cb_person_cred_hist_length'] + 1e-5)
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1e-5)


    return df



def preprocess_train(file_path: str, balance_classes: bool = False, use_feature_eng: bool = False):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['id'])

    if use_feature_eng:
        df = feature_engineering(df)

    df = pd.get_dummies(df, columns=[
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ], drop_first=True)

    df = df.dropna()

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # Divide treino/validação antes de qualquer escala ou balanceamento
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    class_weights_dict = None  # Default

    if balance_classes:
       
       
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(zip(np.unique(y_train), class_weights))

    # Agora escala os dados (separadamente treino e validação!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if balance_classes:
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler, X.columns, class_weights_dict
    else:
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler, X.columns


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
