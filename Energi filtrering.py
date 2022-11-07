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
    dat0 = np.loadtxt(folder + "Compton_spredning " + name + "_ch000_timefiltered" + file_type)
    dat1 = np.loadtxt(folder + "Compton_spredning " + name + "_ch001_timefiltered" + file_type)
    time = dat0[:,0]
    ch0 = dat0[:,1]
    ch1 = dat1[:,1]
    return dat0, dat1

def chToEnergy(ch):
    return 1.121651*ch-18.177

dats = []
angles = [40, 60, 80, 100, 116]
for i in angles:
    theta = i
    i = str(i)
    dat0, dat1 = loadData(i)

    newDat0 = np.array([])
    newDat1 = np.array([])

    for j in range(len(dat0[:,1])):
        j = int(j)

        if j == 0:
            newDat0 = np.array([[dat0[j, 0], chToEnergy(dat0[j, 1])]])
        else:
            newDat0 = np.append(newDat0, [[dat0[j, 0], chToEnergy(dat0[j, 1])]], axis=0)

    for j in range(len(dat1[:, 1])):
        j = int(j)

        if j == 0:
            newDat1 = np.array([[dat1[j, 0], chToEnergy(dat1[j, 1])]])
        else:
            newDat1 = np.append(newDat1, [[dat1[j, 0], chToEnergy(dat1[j, 1])]], axis=0)
    dat0, dat1 = newDat0, newDat1
    toDelete = np.where((dat0[:, 1]+dat1[:, 1]) < 590)
    #print(np.where(dat1[:,1]<0))
    #rint(dat0[713])
    #print(dat1[713])

    toDelete = np.append(toDelete, np.where((dat0[:, 1]+dat1[:, 1]) > 730))

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

