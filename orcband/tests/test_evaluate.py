import sys
sys.path.append("../ml_model/code")
import evaluation
import numpy as np

df1 = np.array([[2],[4],[6]])
df2 = np.array([[3],[5],[7]])
def test_mean_square_error():
    a = evaluation.mean_square_error(df1,df2)
    assert a == 1, "Error: The function of calculating MSE is wrong!"
    return

def test_mean_absolute_error():
    b = evaluation.mean_absolute_error(df1,df2)
    assert b == 1, "Error: The function of calculating MAE is wrong!"
    return

def test_mean_absolute_percentage_error():
    c = evaluation.mean_absolute_percentage_error(df1,df2)
    assert round(c,2) == 0.31, "Error: The function of calculating MAPE is wrong!"
    return

def test_r2_score():
    d = evaluation.r2_score(df1,df2)
    assert d == 0.625, "Error: The function of calculating R2 score is wrong!"
    return
    
