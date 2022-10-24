import glob
import os
import csv

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_time_counts(angle):
    angle = str(angle)
    data_0 = np.loadtxt("Co\Comptorn spredning "+ angle + "_ch000.txt", skiprows=4)
    data_1 = np.loadtxt("Co\Comptorn spredning "+ angle + "_ch001.txt", skiprows=4)
    #data fra 000
    time_in_02us = data_0[:,0]
    count_channel_0 = data_0[:,1]
    #data fra 001
    count_channel_1 = data_1[:,1]

    time = (time_in_02us[-1] - time_in_02us[0])*(10**(-8))
    print((time_in_02us[-1]- time_in_02us[0])*(10**(-8)))
    counts_0 = np.unique(count_channel_0[count_channel_0 > -100], return_counts=True)
    counts_1 = np.unique(count_channel_1[count_channel_1 > -100], return_counts=True)

    
    return time, counts_0[0], counts_0[1], counts_1[0], counts_1[1]

def energy_fuc(x):
    a = 1.1216620315946357
    b = -18.18580680037887
    return a*x+b

angles = [40, 60, 80, 100, 116]
deg40 = get_time_counts(angles[0])
plt.scatter(deg40[1], deg40[2])
plt.show()


