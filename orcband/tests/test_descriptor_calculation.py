import sys
sys.path.append("../datamining/code/")
import sms_dcp_csv
import numpy as np
import pandas as pd

def test_sms_dcp_csv():
    """This function is used to test sms_dcp_csv function"""
    data = pd.read_csv('C:/Users/meng1/Desktop/data_science/HCEPDB_moldata/HCEPDB_moldata.csv')
    data_sample = data.head(10)
    result = sms_dcp_csv.sms_dcp(data_sample)
    
    assert len(result) == 10, "Error: The function don't calculate all input SMILES string"
    assert result.shape[1] == 51, "Error: The function don't select the descriptors based on pearson correlation rightly"
    assert result.columns[0] is 'e_gap_alpha', "Error: The final dataframe isn't right"
    return
