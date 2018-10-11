import math
import numpy as np
import numpy.random as rand
from numpy import ndarray as nd
import statistics
import pandas as pd

def create_chuncks(table):
        table = table.sample(frac=1)       
        trains = []
        tests = []
        n = table.shape[0]    
        for i in range(5):
            begin = int(i * n / 5)
            end = int((i + 1) * n / 5) - 1
            tests.append(table[begin:end])
            dropped_indexes = range(begin, end)
            trains.append(table.drop(table.index[dropped_indexes]))
        return trains, tests 
    
def normalize(X):
        for col in range(X.shape[1]):
            if(pd.unique(X.iloc[:,col].values).size > 2):
                avg = np.mean(X.iloc[:,col])
                var = np.std(X.iloc[:,col])
                X.iloc[:,col] = (X.iloc[:,col] - avg) / var
        return X

def cross_validation(data, learning_rate=0.00005, nsteps=3000, e=0.000000001, weight_low=0, weight_high=1, kweigths=1):        
        results = []
        X = normalize(data)
        trains, tests = create_chuncks(X)    
        
        X_trains = []
        y_trains = []

        y_tests = []   
        X_tests = []
                  
        for i in range(5):                     
            y_trains.append(trains[i].iloc[:,trains[i].shape[1] - 1])       
            X_trains.append(trains[i].drop('Target', axis=1))
            y_tests.append(tests[i].iloc[:,tests[i].shape[1] - 1])       
            X_tests.append(tests[i].drop('Target', axis=1))

            m = X_trains[i].shape[0]
            n = X_trains[i].shape[1]
            W = []
            
            X = np.array(X_trains[i])
            y_train = np.array(y_trains[i])

            for j in range(1,m):          
                X1 = X[j - 1:j].T.reshape(n, 1)
                W = np.random.randint(low = -100, high = 100, size = (1, n))
                y_pred = np.dot(W, X1)
                cost0 = math.fabs(y_train[j - 1] - y_pred)
                k = 0
                while True:
                    dy = float(y_pred - y_train[j - 1])
                    W_tmp = W
                    s = np.dot(dy, X1)   
                    
                    # Gradient descent step
                    dW = 2 * learning_rate * s / m
                    W = W - dW.T
                    y_pred = np.dot(W, X1)
                    cost1 = math.fabs(y_train[j - 1] - y_pred)
                    
                    k += 1
                    
                    if (cost1 > cost0):
                        W = W_tmp
                        break 
                    
                    if ((cost0 - cost1) < e) or (k == nsteps):
                        break
                    
                    cost0 = cost1
                    if(learning_rate - 0.000001 > 0):
                        learning_rate -= 0.000001
            results.append(W)

        return results, y_trains, X_trains, y_tests, X_tests

def predict(X, W):
        n = X.shape[0]
        X1 = np.array(X)
        res = []
        for i in range(n):
            res.append(np.dot(W, X1[i]))
        return res