from logisticRegressionModel import logisticRegressionTrain,logisticRegressionTest


def logisticRegression(trainFile_path:str,testFile_path:str,classWeighbool:bool,use_smote:bool):
    model, scaler, columns = logisticRegressionTrain(trainFile_path, classWeighbool, use_smote)
    logisticRegressionTest(testFile_path,model,scaler,columns)


def main():
    testFile_path:str = '../data/test.csv'  
    trainFile_path:str = '../data/train.csv'  

    logisticRegression(trainFile_path,testFile_path,False,False)

if __name__ == '__main__':
    main()
