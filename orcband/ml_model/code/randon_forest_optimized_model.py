import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from evaluate import r2_score
from evaluate import mean_square_error
from evaluate import mean_absolute_error


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

    # Visualising the Random Forest Regression results
    # in higher resolution and smoother curve
    font1 = {
           'weight': 'normal',
           'size': 10,
           }
    RFR_score = round(r2_score(y_test, Ytest_Pred), 2)
    MAE = round(mean_absolute_error(y_test, Ytest_Pred), 2)
    MSE = round(mean_square_error(y_test, Ytest_Pred), 2)
    rows = ['MSE', 'MAE', '$R^2$']
    cell_text = [['MSE', MSE], ['MAE', MAE], ['$R^2$', RFR_score]]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, Ytrain_Pred, color='blue', label='Train')
    plt.scatter(y_test, Ytest_Pred, color='red', label='Test')
    plt.plot([0, 4], [0, 4], lw=4, color='black', label='Y=X')
    plt.legend(loc='upper left', prop=font1)
    plt.title('$Random \ Forest \ Regression$')
    plt.xlabel('$<E_g> \ Actual \ [eV]$')
    plt.ylabel('$<E_g> \ Predict \ [eV]$')
    my_table = plt.table(cellText=cell_text, colWidths=[0.15]*2, colLoc='center',
                         cellLoc='center', loc='lower right',
                         bbox=[0.75, 0.05, 0.2, 0.2])
    my_table.set_fontsize(20)
    return RFR_score, MAE, MSE
