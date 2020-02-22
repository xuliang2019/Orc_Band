import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
from evaluation import r2_score


def Multiple_Linear_Model(data):
    """Model for Multiple_Linear_Regression, Return a Figure"""
    # Choose the Predictors
    X = data[['AXp-0d', 'AXp-1d', 'AXp-2d',
              'ETA_eta_L', 'ETA_epsilon_3']].values
    Y = data[['e_gap_alpha']].values
    X_train, X_test, Y_train, Y_test = train_test_split(
                                        X, Y, test_size=.25, random_state=1234)

    # Model
    MLR = linear_model.LinearRegression()
    MLR.fit(X_train, Y_train)
    Test_Predictions = MLR.predict(X_test)
    Train_Predictions = MLR.predict(X_train)
    MLRscore = r2_score(Y_test, Test_Predictions)

    # Plot the figure
    figure = plt.figure()
    plt.scatter(Y_train, Train_Predictions, color='blue')
    plt.scatter(Y_test, Test_Predictions, color='r')
    plt.plot([0, 4], [0, 4], lw=4, color='black')
    plt.title('$Multiple \ Linear \ Regression$')
    plt.xlabel('$<Eg> \ Actual \ [eV]$')
    plt.ylabel('$<Eg> \ Predict \ [eV]$')
    plt.show()

    print('The score of this regression in this case is: ', MLRscore)
    return MLRscore


def Polynomial_Model(data):
    """Model for Polynomial Regression Model"""
    # Choose the Predictors
    X = data[['AXp-0d', 'AXp-1d', 'AXp-2d',
              'ETA_eta_L', 'ETA_epsilon_3']].values
    Y = data[['e_gap_alpha']].values
    X_train, X_test, Y_train, Y_test = train_test_split(
                                        X, Y, test_size=.25, random_state=1234)

    # Model
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)  # transfer to normal distribution
    X_test = sc_X.transform(X_test)
    L_R = LinearRegression()
    Poly_Reg = PolynomialFeatures(degree=4)
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
    plt.show()
    print('The score of this regression in this case is: ', PLMscore)

    return PLMscore


def Random_Forest_Model(data):  # Pending
    """Model for Random Forest Model"""
    # Choose the Predictors
    X = data[['AXp-0d', 'AXp-1d', 'AXp-2d',
              'ETA_eta_L', 'ETA_epsilon_3']].values
    Y = data[['e_gap_alpha']].values
    X_train, X_test, Y_train, Y_test = train_test_split(
                                        X, Y, test_size=.25, random_state=1234)

    # Model
    regressor = RandomForestRegressor(n_estimators=300, random_state=0,
                                      min_samples_split=15)
    regressor.fit(X_train, Y_train)

    Ytrain_Pred = regressor.predict(X_train)
    Ytest_Pred = regressor.predict(X_test)
    RFMscore = r2_score(Y_test, Ytest_Pred)

    # Figure
    figure = plt.figure()
    plt.scatter(Y_train, Ytrain_Pred, color='blue')
    plt.scatter(Y_test, Ytest_Pred, color='red')
    plt.plot([0, 4], [0, 4], lw=4, color='black')
    plt.title('$Random \ Forest \ Regression$')
    plt.xlabel('$<Eg> \ Actual \ [eV]$')
    plt.ylabel('$<Eg> \ Predict \ [eV]$')

    # Score of the Model
    print('The score of this regression in this case is: ',
          RFMscore)

    return RFMscore



def neural_network_model(data):
    """Neural net work model"""
    X = data[['AXp-0d', 'AXp-1d', 'AXp-2d', 'ETA_eta_L',
              'ETA_epsilon_3']].values
    Y = data[['e_gap_alpha']].values
    X_train_pn, X_test_pn, y_train, y_test = train_test_split(X, Y,
                                                              test_size=0.25,
                                                              random_state=1234
                                                              )
    # create the scaler from the training data only and keep it for later use
    X_train_scaler = StandardScaler().fit(X_train_pn)
    # apply the scaler transform to the training data
    X_train = X_train_scaler.transform(X_train_pn)
    X_test = X_train_scaler.transform(X_test_pn)

    def neural_model():
        # assemble the structure
        model = Sequential()
        model.add(Dense(5, input_dim=5, kernel_initializer='normal',
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(8, kernel_initializer='normal', activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(20, kernel_initializer='normal', activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
        # model.add(Dense(4, kernel_initializer='normal',activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    # initialize the andom seed as this is used to generate
    # the starting weights
    np.random.seed(1234)
    # create the NN framework
    estimator = KerasRegressor(build_fn=neural_model,
                               epochs=1200, batch_size=25000, verbose=0)
    history = estimator.fit(X_train, y_train, validation_split=0.33,
                            epochs=1200, batch_size=10000, verbose=0)
    print("final MSE for train is %.2f and for validation is %.2f" %
          (history.history['loss'][-1], history.history['val_loss'][-1]))
    plt.figure(figsize=(5, 5))
    prediction = estimator.predict(X_test)
    plt.scatter(y_train, estimator.predict(X_train), color='blue')
    plt.scatter(y_test, prediction, color='red')
    plt.plot([0, 4], [0, 4], lw=4, color='black')
    plt.title('$Neural \ Network \ Model$')
    plt.xlabel('$<Eg> \ Actual \ [eV]$')
    plt.ylabel('$<Eg> \ Predict \ [eV]$')
    return r2_score(y_test, prediction)