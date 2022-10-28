import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
folder = ''
seperator = ' '
file_type = '.txt'


def loadData(name):
    dat0 = np.loadtxt(folder + "filtered_BGO" + name + file_type)
    dat1 = np.loadtxt(folder + "filtered_NaI" + name + file_type)
    time = dat0[:,0]
    ch0 = dat0[:,1]
    ch1 = dat1[:,1]
    return [time, ch0, ch1]

dats = []
angles = [40, 60, 80, 100]
for i in angles:
    i = str(i)
    dat = [loadData(i)]
    dats += dat
    plt.plot(dat[0][1], dat[0][2])
    plt.title("BGO vs NaI " + str(i) + " degrees")
    plt.xlabel("BGO")
    plt.ylabel("NaI")
    plt.show()