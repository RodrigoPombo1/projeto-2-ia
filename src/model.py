from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from parser import preprocess_train, preprocess_test
import pandas as pd
from xgboost import XGBClassifier
from collections import Counter
import time

import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from xgboost import XGBClassifier

def train_model(trainFile_path: str, model, classWeightbool: bool = False, use_feature_eng: bool = False):
    if classWeightbool:
        # preprocess_train retorna class_weights quando balance_classes=True
        X_train, X_val, y_train, y_val, scaler, columns, class_weights = preprocess_train(
            trainFile_path, balance_classes=True, use_feature_eng=use_feature_eng
        )
        # Ajusta parâmetros do modelo para classes desbalanceadas
        if isinstance(model, XGBClassifier):
            counter = Counter(y_train)
            weight = counter[0] / counter[1]
            model.set_params(scale_pos_weight=weight)
        elif hasattr(model, 'class_weight') and model.class_weight is None:
            model.class_weight = class_weights
    else:
        # preprocess_train não retorna class_weights quando balance_classes=False
        X_train, X_val, y_train, y_val, scaler, columns = preprocess_train(
            trainFile_path, balance_classes=False, use_feature_eng=use_feature_eng
        )
        class_weights = None  # para consistência
    
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    start_pred = time.time()
    y_pred = model.predict(X_val)
    end_pred = time.time()

    print("Exatidão (validação):", accuracy_score(y_val, y_pred))
    print("\nClassification report:\n", classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    print("Matriz de confusão:")
    print(cm)

    print(f"Tempo de treino: {end_train - start_train:.4f} segundos")
    print(f"Tempo de predição (validação): {end_pred - start_pred:.4f} segundos")

    return model, scaler, columns


def test_model(testFile_path: str, model, scaler, columns, use_feature_eng: bool = False):
    start_test = time.time()
    X_test, ids = preprocess_test(testFile_path, scaler, columns, use_feature_eng=use_feature_eng)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        result = y_proba
    else:
        y_pred = model.predict(X_test)
        result = y_pred
    end_test = time.time()

    result_df = pd.DataFrame({
        'id': ids,
        'loan_status': result
    })

    result_df.to_csv("submission.csv", index=False)
    print("Submission saved on 'submission.csv'")
    print(f"Tempo para processar teste: {end_test - start_test:.4f} segundos")
