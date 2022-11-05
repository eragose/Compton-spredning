import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
folder = 'Co/'
seperator = ' '
file_type = '.txt'


def loadData(name):
    dat0 = np.loadtxt(folder + "Energy_filtered Compton_spredning " + name + "_ch000" + file_type, skiprows=4)
    dat1 = np.loadtxt(folder + "Energy_filtered Compton_spredning " + name + "_ch001" + file_type, skiprows=4)
    return dat0, dat1

dat0, dat1 = loadData('80')
NaIChannels, NaICounts = np.unique(dat1[:,1], return_counts=True)
plt.plot(NaIChannels, NaICounts)
plt.show()