import pandas as pd
import numpy as np
from multiprocessing import freeze_support
from rdkit import Chem
from mordred import Calculator, descriptors


def sms_dcp(data):
    """This is a function to output the descriptors
        which has more than 0.65 correlation with bandgap"""

    SMLSTR_Data = np.array(data['SMILES_str'])
    # Transfer array to list
    SMLSTR_Data = SMLSTR_Data.tolist()
    BandGap_Data = np.array(data['e_gap_alpha'])
    BandGap_Data = BandGap_Data.tolist()

    freeze_support()
    mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES_str']]
    # Create Calculator
    calc = Calculator(descriptors)
    # map method calculate multiple molecules (return generator)
    # pandas method calculate multiple molecules (return pandas DataFrame)
    raw_data = calc.pandas(mols)
    # Insert SMILES String in to raw_data
    raw_data.insert(0, 'SMILES_str', SMLSTR_Data)
    raw_data.insert(1, 'e_gap_alpha', BandGap_Data)

    Correlation_BG = raw_data.corr()
    # Save as DataFrame
    Correlation_BG = pd.DataFrame(data=Correlation_BG)
    # Create a DataFrame
    DCPdata = pd.DataFrame()

    for index in range(len(Correlation_BG)):
        if Correlation_BG['e_gap_alpha'][index] >= 0.65:
            indexX = Correlation_BG.index[index]
            values = raw_data[indexX]
            DCPdata[indexX] = values
        elif Correlation_BG['e_gap_alpha'][index] <= -0.65:
            indexX = Correlation_BG.index[index]
            values = raw_data[indexX]
            DCPdata[indexX] = values

    # Save as CSV File
    DCPdata.to_csv('DCPdata.csv')

    return DCPdata
