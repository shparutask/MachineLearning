import LinRegression as lr
import metrics as m
import pandas as pd
import numpy as np
import random

FILENAME_READ = "../Dataset/Training/dataset.csv"
FILENAME_WRITE = "../Dataset/Training/table.csv"

if __name__ == "__main__":    
    data = pd.read_csv(FILENAME_READ)
    data = data.drop('Unnamed: 0', axis=1)
    columns = data.columns

    weights, y_trains, X_trains, y_tests, X_tests = lr.cross_validation(data)
    
    data_write = pd.DataFrame(columns=["","T1", "T2", "T3", "T4", "T5", "E", "STD"])
    
    for i in range(5):
        y_pred_test = lr.predict(X_tests[i], weights[i])
        y_pred_train = lr.predict(X_trains[i], weights[i])
        y_test = np.array(y_tests[i])
        y_train = np.array(y_trains[i])

        r2_test = m.R2(y_test, y_pred_test)
        r2_train = m.R2(y_train, y_pred_train)
        rmse_test = m.RMSE(y_test, y_pred_test)
        rmse_train = m.RMSE(y_train, y_pred_train)
        
        data_write["T" + str(i+1)] = [r2_test, r2_train, rmse_test, rmse_train] + list(weights[i].reshape(weights[i].shape[0], 1))
        
    data_write["E"] = data_write[["T1", "T2", "T3", "T4", "T5"]].mean(axis=1)
    data_write["STD"] = data_write[["T1", "T2", "T3", "T4", "T5"]].std(axis=1)

    data_write.index = ["R^2_test", "R^2_train", "RMSE_test", "RMSE_train"] + list(columns)
    data_write.to_csv("result.csv")