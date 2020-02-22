# import package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from evaluation import r2_score


def Multiple_Linear_Model(data):
    X = data[['AXp-0d', 'AXp-1d', 'AXp-2d',
              'ETA_eta_L', 'ETA_epsilon_3']].values
    y = data[['e_gap_alpha']].values  # load data
    # split tran and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                        random_state=1234)

    # build multiple linear model
    MLR = linear_model.LinearRegression()
    MLR.fit(X_train, y_train)
    testpred = MLR.predict(X_test)
    trainpred = MLR.predict(X_train)
    MLR_score = r2_score(y_test, testpred)
    plt.scatter(y_train, trainpred, color='blue')
    plt.scatter(y_test, testpred, color='r')
    plt.plot([0, 4], [0, 4], lw=4, color='black')
    plt.title('$Multiple \ Linear \ Regression$')
    plt.xlabel('$<Eg> \ Actual \ [eV]$')
    plt.ylabel('$<Eg> \ Predict \ [eV]$')
    return MLR_score
