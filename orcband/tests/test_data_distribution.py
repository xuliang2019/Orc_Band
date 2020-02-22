import sys
sys.path.append("../datamining/code/")
import data_distribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_datadistribution():
    data = pd.read_scv('../../documentation/data/DescriptorsDataset.csv')
    data_distribution.datadistribution(data['e_gap_alpha'])
    xname = plt.gca().get_xlabel()
    yname = plt.gca().get_ylabel()
    title = plt.gca().get_title()
    assert title == 'The Distribution of Bandgap', 'Error: The title of figure is wrong'
    assert yname == '$Counts$','Error: The y label is not Counts'
    assert xname == '$<E_g> \\ [eV]$', 'Error: The x label is not right'
    return
