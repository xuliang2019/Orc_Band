import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def datadistribution(data):
    """This function is used to plot the distribution of bandgap dataset"""
    plt.figure(figsize=(8, 6))
    plt.hist(data['e_gap_alpha'], rwidth=0.9, bins=20)
    plt.xlabel('$<E_g> \ [eV]$', fontsize=18)
    plt.ylabel('$Counts$', fontsize=18)
    plt.title('The Distribution of Bandgap', fontsize=18)
    plt.savefig('The distribution of band', dpi=80)
    return
