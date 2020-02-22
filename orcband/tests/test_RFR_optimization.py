import sys
sys.path.append("../ml_model/code")
import rfr_optimization

def test_rfr_optimization():
    result = rfr_optimization.search_hyperparameter()
    assert result is None, "Wrong!"
    return
