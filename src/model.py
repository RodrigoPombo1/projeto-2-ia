from sklearn.metrics import accuracy_score, classification_report
from parser import preprocess_train, preprocess_test
import pandas as pd
from xgboost import XGBClassifier
from collections import Counter

def train_model(trainFile_path: str, model, classWeighbool: bool = False, use_smote: bool = False, use_feature_eng: bool = False):
    if classWeighbool:
        X_train, X_val, y_train, y_val, scaler, columns, class_weights = preprocess_train(
            trainFile_path, balance_classes=True, use_smote=use_smote, use_feature_eng=use_feature_eng
        )
        # Se o modelo for XGBClassifier e classWeighbool for True, aplica scale_pos_weight 
        if isinstance(model, XGBClassifier):
            counter = Counter(y_train)
            weight = counter[0] / counter[1]
            model.set_params(scale_pos_weight=weight)

        # Se o modelo suportar class_weight e ainda não tiver definido
        elif hasattr(model, 'class_weight') and model.class_weight is None:
            model.class_weight = class_weights

    else:
        X_train, X_val, y_train, y_val, scaler, columns = preprocess_train(
            trainFile_path, balance_classes=False, use_smote=use_smote, use_feature_eng=use_feature_eng
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print("exatidão (validação):", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    return model, scaler, columns


def test_model(testFile_path: str, model, scaler, columns, use_feature_eng: bool = False):
    X_test, ids = preprocess_test(testFile_path, scaler, columns, use_feature_eng=use_feature_eng)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        result = y_proba
    else:
        y_pred = model.predict(X_test)
        result = y_pred

    result_df = pd.DataFrame({
        'id': ids,
        'loan_status': result
    })

    result_df.to_csv("submission.csv", index=False)
    print("Submission saved on 'submission.csv'")
