import sys
sys.path.append("../datamining/code/")
import lasso_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_lassoreg():
    """This function is to test lasso regression code"""
    data=pd.read_csv('../../documentation/data/DescriptorsDataset.csv')
    data100=data.sample(frac=0.01)
    lasso_regression.lassoreg(data100)
    title = plt.gcf().get_axes()
    assert str(title[1].title) == "Text(0.5, 1.0, 'error vs $\\\\lambda$')",'The name of second subplot is not right'
    assert str(title[0].title) == "Text(0.5, 1.0, 'RR coefs vs $\\\\lambda$')", 'The name of firsr subplot is not right'
    return
