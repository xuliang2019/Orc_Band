import numpy as np
import pandas as pd
import pickle

from sklearn import linear_model
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from multiprocessing import freeze_support
from rdkit import Chem
from mordred import Calculator, descriptors

def load_model():
    """
    load the trained prediction model
    """
    with open('../model/regressor.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

def sms_bandgap(sms):
    """Function from sms to predict Bandgap,
       sms represents the smiles string of the chemical that you want to predict,
       >>> sms_bandgap('c1ccccc1')
       >>> array([2.70371115])
    """
    bandgap = pd.DataFrame(columns = ['substance','bandgap'])
    bandgap.loc[0,'substance'] = sms
    freeze_support()   
    mols = Chem.MolFromSmiles(sms) #transform smiles string to molecular structure
    if mols is None:
        raise TypeError('Invalid Smiles String')
    else:
        m = [Chem.MolFromSmiles(sms)]
        calc = Calculator(descriptors)
        raw_data = calc.pandas(m) #calculate descriptors
        new = {'AXp-0d': raw_data['AXp-0d'].values,
               'AXp-1d': raw_data['AXp-1d'].values,
               'AXp-2d': raw_data['AXp-2d'].values,
               'ETA_eta_L': raw_data['ETA_eta_L'].values,
               'ETA_epsilon_3': raw_data['ETA_epsilon_3'].values}  # extract the five most useful descriptors data
        new_data = pd.DataFrame(index=[1], data=new)
        regressor2 = load_model()
        bandgap.loc[0,'bandgap'] = regressor2.predict(new_data)[0]  # calculate bandgap
        return bandgap

