import sys
sys.path.append("../ml_model/code")
import machine_learning_model
import pandas as pd

def test_multiple_linaer_model():
    data = pd.read_csv('../../documentation/data/DescriptorsDataset.csv')
    score = machine_learning_model.Multiple_Linear_Model(data)
    assert score < 0.6, "should be 0.57"
    return

def test_Polynomial_Model():
    data = pd.read_csv('../documentation/data/DescriptorsDataset.csv')
    score1 = machine_learning_model.Polynomial_Model(data)
    assert score1 < 0.65, "should be less than 0.65"
    return

def test_Random_Forest_Model():
    data = pd.read_csv('../documentation/data/DescriptorsDataset.csv')
    score2 = machine_learning_model.Random_Forest_Model(data)
    assert round(score2,2) == 0.7, "Error: The function is not right"
    return

def test_Neural_Network_Model():
    data = pd.read_csv('../documentation/data/DescriptorsDataset.csv')
    score3 = machine_learning_model.Neural_Network_Model(data)
    assert score3 < 0.6, "should be about 0.56"
    return
