import numpy as np
import math

def RMSE(y_test, y_pred):
        sum = 0
        N = len(y_pred)
        for i in range(N):
            sum += float(y_test[i] - y_pred[i])**2
        return math.sqrt(sum/N)
                
def R2(y_test, y_pred):
        sum1 = 0
        sum2 = 0
        avg = np.average(y_test)
        for i in range(len(y_test)):
            sum1 = (float(y_pred[i]) - float(y_test[i]))**2
            sum2 = (float(y_test[i]) - float(avg))**2
        if (sum2 == 0): return 1
        return 1 - sum1 / sum2
     