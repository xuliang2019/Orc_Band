import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from evaluation import r2_score


def Random_Forest_Reg(data):
    # import data
    X = data[['AXp-0d', 'AXp-1d', 'AXp-2d', 'ETA_eta_L',
              'ETA_epsilon_3']].values
    y = data[['e_gap_alpha']].values
    # split test and train dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(),
                                                        test_size=.25,
                                                        random_state=1234)
    regressor = RandomForestRegressor(n_estimators=300, random_state=123,
                                      min_samples_split=15)
    regressor.fit(X_train, y_train.ravel())
    # Predicting a new result with the Random Forest Regression
    Ytrain_Pred = regressor.predict(X_train)
    Ytest_Pred = regressor.predict(X_test)
    RFR_score = r2_score(y_test, Ytest_Pred)

    # Visualising the Random Forest Regression results
    # in higher resolution and smoother curve
    plt.scatter(y_train, Ytrain_Pred, color='blue')
    plt.scatter(y_test, Ytest_Pred, color='red')
    plt.plot([0, 4], [0, 4], lw=4, color='black')
    plt.title('$Random \ Forest \ Regression$')
    plt.xlabel('$<Eg> \ Actual \ [eV]$')
    plt.ylabel('$<Eg> \ Predict \ [eV]$')
    return RFR_score
