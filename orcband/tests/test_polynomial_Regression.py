import sys
sys.path.append("../ml_model/code")
import polynomial_regression_model
import pandas as pd
import matplotlib.pyplot as plt

def test_polynomial_Reg():
    """This function is to test polynomial regression code"""
    data = pd.read_csv('../../documentation/data/DescriptorsDataset.csv')
    score = polynomial_regression_model.Polynomial_Model(data)
    xname = plt.gca().get_xlabel()
    yname = plt.gca().get_ylabel()
    title = plt.gca().get_title()
    xscale = plt.gca().get_xlim()
    yscale = plt.gca().get_ylim()
    assert round(score,2) == 0.7, "Error: The code for r2 score is wrong!"
    assert xname == '$<Eg> \\ Actual \\ [eV]$', "The figure x label is wrong!"
    assert yname == '$<Eg> \\ Predict \\ [eV]$', "The figure y label is wrong!"
    assert title == '$Polynomial \\ Regression$', "The figure title is wrong!"
    assert xscale[0] == -0.2 and xscale[1] == 4.2, "The range of x is wrong!"
    assert yscale[0] == -0.2 and yscale[1] == 4.2, "The range of y is wrong!"
    return
