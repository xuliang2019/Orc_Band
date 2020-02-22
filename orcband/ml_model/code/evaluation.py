import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def mean_square_error(y_test, ytest_Pred):
    '''calculate mean square error'''
    return np.mean((y_test - ytest_Pred)**2)


def mean_absolute_error(y_test, ytest_Pred):
    '''calculate mean absolute error'''
    return np.mean(abs(y_test - ytest_Pred))


def mean_absolute_percentage_error(y_test, ytest_Pred):
    '''calculate mean absolute percentage error'''
    return np.mean(abs(y_test - ytest_Pred)/y_test)


def r2_score(y_test, ytest_Pred):
    '''calculate R2'''
    return r2_score(y_test, ytest_Pred)
