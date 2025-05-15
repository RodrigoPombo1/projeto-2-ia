from logisticRegressionModel import logisticRegressionTrain,logisticRegressionTest

def main():
    testFile_path:str = '../data/test.csv'  
    trainFile_path:str = '../data/train.csv'  
    model, scaler, columns = logisticRegressionTrain(trainFile_path, classWeighbool=False, use_smote=False)
    logisticRegressionTest(trainFile_path,model,scaler,columns)


    """ logisticRegressionTrain(trainFile_path, classWeighbool=True, use_smote=True)
    logisticRegressionTrain(trainFile_path, classWeighbool=True, use_smote=False) """


if __name__ == '__main__':
    main()
