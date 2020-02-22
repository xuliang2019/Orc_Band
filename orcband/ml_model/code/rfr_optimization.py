import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier


def search_hyperparameter(data):
    X = data[['AXp-0d', 'AXp-1d', 'AXp-2d', 'ETA_eta_L',
              'ETA_epsilon_3']].values
    y = data[['e_gap_alpha']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(),
                                                        test_size=.25,
                                                        random_state=1234)
    param_dist = {"max_depth": [3, None],
                  "min_samples_split": sp_randint(2, 30),
                  "bootstrap": [True, False]}
    n_iter_search = 20
    rfr = RandomForestRegressor(n_estimators=300, random_state=123)
    random_search = RandomizedSearchCV(rfr, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)
    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    results = random_search.cv_results_
    for i in range(1, 6):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return
