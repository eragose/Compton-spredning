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
    dat0 = np.loadtxt(folder + "Compton_spredning " + name + "_ch000_timefiltered" + file_type, skiprows=4)
    dat1 = np.loadtxt(folder + "Compton_spredning " + name + "_ch001_timefiltered" + file_type, skiprows=4)
    time = dat0[:,0]
    ch0 = dat0[:,1]
    ch1 = dat1[:,1]
    return dat0, dat1

dats = []
angles = [40, 60, 80, 100, 116]
for i in angles:
    theta = i
    i = str(i)
    dat0, dat1 = loadData(i)
    toDelete = np.where((dat0[:, 1]+dat1[:, 1]) < 550)
    #print(np.where(dat1[:,1]<0))
    #rint(dat0[713])
    #print(dat1[713])

    toDelete = np.append(toDelete, np.where((dat0[:, 1]+dat1[:, 1]) > 700))

    dat0 = np.delete(dat0, toDelete, 0)
    dat1 = np.delete(dat1, toDelete, 0)
    dats += [(dat0, dat1)]

    plt.scatter(dat0[:, 1], dat1[:, 1], alpha=0.05)

    plt.title("BGO vs NaI " + str(i) + " degrees")
    plt.xlabel("BGO")
    plt.ylabel("NaI")
    plt.show()
    np.savetxt(folder + 'Energy_filtered Compton_spredning ' + i + '_ch000.txt', dat0, delimiter=" ", fmt='%s')
    np.savetxt(folder + 'Energy_filtered Compton_spredning ' + i + '_ch001.txt', dat1, delimiter=" ", fmt='%s')
