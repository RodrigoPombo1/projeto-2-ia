import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def parse_csv(file_path: str) -> None:
    df = pd.read_csv(file_path)

    df = df.drop(columns=['id'])

    df = pd.get_dummies(df, columns=[
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ], drop_first=True)

    df = df.dropna()

    X = df.drop(columns=['loan_int_rate'])
    y = df['loan_int_rate']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
