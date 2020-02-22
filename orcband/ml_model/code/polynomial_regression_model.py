import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from evaluation import r2_score


def Polynomial_Model(data):
    """Model for Polynomial Regression Model"""
    X = data[['AXp-0d', 'AXp-1d', 'AXp-2d',
              'ETA_eta_L', 'ETA_epsilon_3']].values
    Y = data[['e_gap_alpha']].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25,
                                                        random_state=1234)

    # Model
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)  # transfer to normal distribution
    X_test = sc_X.transform(X_test)
    L_R = LinearRegression()
    Poly_Reg = PolynomialFeatures(degree=2)
    X_poly = Poly_Reg.fit_transform(X_train)
    Poly_Reg.fit(X_poly, Y_train)
    L_R.fit(X_poly, Y_train)
    Ytrain_Pred = L_R.predict(Poly_Reg.fit_transform(X_train))
    Ytest_Pred = L_R.predict(Poly_Reg.fit_transform(X_test))
    PLMscore = r2_score(Y_test, Ytest_Pred)

    # Figure
    figure = plt.figure()
    plt.scatter(Y_train, Ytrain_Pred, color='blue')
    plt.scatter(Y_test, Ytest_Pred, color='red')
    plt.plot([0, 4], [0, 4], lw=4, color='black')
    plt.title('$Polynomial \ Regression$')
    plt.xlabel('$<Eg> \ Actual \ [eV]$')
    plt.ylabel('$<Eg> \ Predict \ [eV]$')
    print('The score of this regression in this case is: ',
          PLMscore)

    return PLMscore
