from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model import train_model,test_model
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from model import train_model, test_model
from xgboost import XGBClassifier

def logisticRegression(trainFile_path: str, testFile_path: str, classWeighbool: bool,use_feature_eng:bool):
    start_time = time.time()

    modelUsed = LogisticRegression(max_iter=1000, random_state=42)
    model, scaler, columns = train_model(trainFile_path, modelUsed, classWeighbool,use_feature_eng)
    test_model(testFile_path, model, scaler, columns,use_feature_eng)

    end_time = time.time()
    print(f"[LogisticRegression] Tempo de execução: {end_time - start_time:.2f} segundos")

def randomForest(trainFile_path: str, testFile_path: str, classWeighbool: bool,use_feature_eng:bool):
        start_time = time.time()

        modelUsed = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced' if classWeighbool else None
        )
        model, scaler, columns = train_model(trainFile_path, modelUsed, classWeighbool,use_feature_eng)
        test_model(testFile_path, model, scaler, columns,use_feature_eng)

        end_time = time.time()
        print(f"[RandomForest] Tempo de execução: {end_time - start_time:.2f} segundos")



def xgboost_model(trainFile_path: str, testFile_path: str, classWeighbool: bool, use_feature_eng: bool):
    start_time = time.time()

    modelUsed = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )

    # Treina e testa o modelo
    model, scaler, columns = train_model(trainFile_path, modelUsed, classWeighbool, use_feature_eng)
    test_model(testFile_path, model, scaler, columns, use_feature_eng)

    end_time = time.time()
    print(f"[XGBoost] Tempo de execução: {end_time - start_time:.2f} segundos")


def main():
    testFile_path: str = '../data/test.csv'  
    trainFile_path: str = '../data/train.csv'  

    randomForest(trainFile_path, testFile_path, False,False)
    randomForest(trainFile_path, testFile_path, True,False)
    randomForest(trainFile_path, testFile_path, False,True)
if __name__ == '__main__':
    main()
