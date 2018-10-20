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
            end = int((i + 1) * n / 5)
            tests.append(table[begin:end])
            dropped_indexes = range(begin, end)
            trains.append(table.drop(table.index[dropped_indexes]))
        return trains, tests 
    
def normalize(data):	
    count_uniq_val = data.apply(lambda x:len(x.unique()))
    normalize_columns = list(count_uniq_val[(count_uniq_val > 2)].index)
    normalize_data = data[normalize_columns]
    data[normalize_columns] = (normalize_data - normalize_data.mean()) / normalize_data.std()
    return data

def gradient_descent_step(X, dy, learning_rate, m, n, W):
    return W - learning_rate * 2 / m * np.dot(dy.T, X).reshape(n, 1)

def linear_regression(X, y_train, learning_rate, n, m, e, nsteps): 
    W = np.random.random_sample(size=(n, 1))
    y_pred = np.dot(X, W).reshape(m, 1)
    cost0 = np.sum((y_pred - y_train) ** 2) / (len(y_train))
    k = 0  
    while True:
        dy = y_pred - y_train
        W_tmp = W
        
        # Gradient descent step
        W = gradient_descent_step(X, dy, learning_rate, m, n, W)
        y_pred = np.dot(X, W)
        cost1 = np.sum((y_pred - y_train) ** 2) / (len(y_train))
        k += 1
        
        if (cost1 > cost0):
            W = W_tmp
            break
        
        if ((cost0 - cost1) < e) or (k == nsteps):
            break
        
        cost0 = cost1
        learning_rate -= e*k
    
    return W


def cross_validation(data, learning_rate=0.05, nsteps=3000, e=0.000000001, weight_low=0, weight_high=1, kweigths=1):        
        results = []
        trains, tests = create_chuncks(data)    
        
        X_trains = []
        y_trains = []

        y_tests = []   
        X_tests = []
                  
        for i in range(5): 
            y_trains.append(trains[i].iloc[:,trains[i].shape[1] - 1])       
            X_trains.append(normalize(trains[i].drop('Target', axis=1)))
            y_tests.append(tests[i].iloc[:,tests[i].shape[1] - 1])       
            X_tests.append(normalize(tests[i].drop('Target', axis=1)))

            m = X_trains[i].shape[0]                        

            X_trains[i] = np.hstack((np.ones(m).reshape(m, 1), X_trains[i]))            
            X_tests[i] = np.hstack((np.ones(X_tests[i].shape[0]).reshape(X_tests[i].shape[0], 1), X_tests[i]))

            n = X_trains[i].shape[1]

            X = np.array(X_trains[i])
            y_train = np.array(y_trains[i]).reshape(m, 1)

            results.append(linear_regression(X, y_train, learning_rate, n, m, e, nsteps))

        return results, y_trains, X_trains, y_tests, X_tests

def predict(X, W):
        n = X.shape[0]
        X1 = np.array(X)
        return np.dot(X1, W)