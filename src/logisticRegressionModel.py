from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from parser import preprocess_train,preprocess_test
import pandas as pd


def logisticRegressionTrain(trainFile_path: str, classWeighbool: bool = False, use_smote: bool = False):
    if classWeighbool:
        X_train, X_val, y_train, y_val, scaler, columns, class_weights = preprocess_train(
            trainFile_path, balance_classes=True, use_smote=use_smote
        )
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights)
    else:
        X_train, X_val, y_train, y_val, scaler, columns = preprocess_train(
            trainFile_path, balance_classes=False, use_smote=use_smote
        )
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print("Acurácia (validação):", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    return model, scaler, columns




def logisticRegressionTest(testFile_path: str, model, scaler, columns):
    X_test, ids = preprocess_test(testFile_path, scaler, columns)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    result_df = pd.DataFrame({
        'id': ids,
        'loan_status': y_proba  
    })
    
    result_df.to_csv("submission.csv", index=False)
    print("Submission saved on 'submission.csv'")
