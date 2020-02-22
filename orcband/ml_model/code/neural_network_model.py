from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from keras import regularizers


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
