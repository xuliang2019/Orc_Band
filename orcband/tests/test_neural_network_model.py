import sys
sys.path.append("../ml_model/code")
import neural_network_model
import pandas as pd
import matplotlib.pyplot as plt

def test_neural_network_model():
    """This function is to test neural network model"""
    data = pd.read_csv('../../documentation/data/DescriptorsDataset.csv')
    score = neural_network_model.neural_network_model(data)
    xname = plt.gca().get_xlabel()
    yname = plt.gca().get_ylabel()
    title = plt.gca().get_title()
    xscale = plt.gca().get_xlim()
    yscale = plt.gca().get_ylim()
    assert round(score,2) == 0.56, "Error: The code for r2 score is wrong!"
    assert xname == '$<Eg> \\ Actual \\ [eV]$', "The figure x label is wrong!"
    assert yname == '$<Eg> \\ Predict \\ [eV]$', "The figure y label is wrong!"
    assert title == '$Polynomial \\ Regression$', "The figure title is wrong!"
    assert xscale[0] == -0.2 and xscale[1] == 4.2, "The range of x is wrong!"
    assert yscale[0] == -0.2 and xscale[1] == 4.2, "The range of y is wrong!"
    return
