import sys
sys.path.append("../ml_model/code")
import random_forest_regression_model
import pandas as pd
import matplotlib.pyplot as plt

def test_Random_Forest_Reg():
    """This function is to test random forest regression code"""
    data = pd.read_csv('../../documentation/data/DescriptorsDataset.csv')
    result = random_forest_regression_model.Random_Forest_Reg(data)
    xname = plt.gca().get_xlabel()
    yname = plt.gca().get_ylabel()
    title = plt.gca().get_title()
    xscale = plt.gca().get_xlim()
    yscale = plt.gca().get_ylim()
    assert round(result,2) == 0.7, "Error: The code for r2 score is wrong!"
    assert xname == '$<Eg> \\ Actual \\ [eV]$', "The figure x label is wrong!"
    assert yname == '$<Eg> \\ Predict \\ [eV]$', "The figure y label is wrong!"
    assert title == '$Random \\ Forest \\ Regression$', "The figure title is wrong!"
    assert xscale[0] == -0.2 and xscale[1] == 4.2, "The range of x is wrong!"
    assert yscale[0] == -0.2 and xscale[1] == 4.2, "The range of y is wrong!"
    
    return
