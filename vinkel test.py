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

test = np.loadtxt(folder + "Comptorn spredning 40_ch000" + file_type, skiprows=4)
#counts = data[:, 1]
#(x, y) = np.unique(counts, return_counts=True)
#lI = np.where(x == 0)[0][0]
#hI = np.where(x >= 1000)[0][0]
#x = x[lI:hI]
#y = y[lI:hI]
#plt.plot(x, y)
#plt.title("test")
#plt.show()
#test = np.loadtxt("Co/Comptorn spredning 40_ch000.dat")
#test = np.fromfile("Co/Comptorn spredning 40_ch000.dat", dtype=int)

def gaussFit(x, mu, sig, a):
    lny = np.log(a) - ((x-mu)**2)/(2*sig)
    return np.exp(lny)+1

#data = np.loadtxt(folder + "compton test " + "3" + "_ch000" + file_type, skiprows=4)
#time = data[:,0]
def loadData(name):
    dat0 = np.loadtxt(folder + "Comptorn spredning " + name + "_ch000" + file_type, skiprows = 4)
    dat1 = np.loadtxt(folder + "Comptorn spredning " + name + "_ch001" + file_type, skiprows = 4)
    time = dat0[:,0]
    ch0 = dat0[:,1]
    ch1 = dat1[:,1]
    return [time, ch0, ch1]

def chToEnergy(ch):
    return 1.12166*ch-18.19

def chsToEnergy(data):
    size = len(data[0])
    for i in np.linspace(0, size-1, size):
        i = int(i)
        data[1][i] = chToEnergy(data[1][i])
        data[2][i] = chToEnergy(data[2][i])

def conservation(E1, theta):
    mc2 = 0.5*1000 #keV
    return E1/(1+(E1/mc2)*(1-np.cos(theta*np.pi/180)))
def checkConservation(E1, E2, theta, error = 0.11):
    if (np.abs(1-E2/conservation(E1, theta)) < error):
        if np.abs(E1-E2)>120:
            return True
    else:
        return False


def getFit(name: str, data: tuple, guess: [int, int, int], lower_limit: int = 0, upper_limit: int = 1000):
    x = data[0]
    y = data[1]
    yler = np.sqrt(y)
    pinit = guess
    xhelp = np.linspace(lower_limit, upper_limit, 500)
    popt, pcov = curve_fit(gaussFit, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
    print(name)
    print('mu :', popt[0])
    print('sigma :', popt[1])
    print('scaling', popt[2])
    #print('background', popt[3], popt[4])
    perr = np.sqrt(np.diag(pcov))
    print('usikkerheder:', perr)
    chmin = np.sum(((y - gaussFit(x, *popt)) / yler) ** 2)
    print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4))

    plt.plot(x, y, color="r", label="data")
    plt.plot(xhelp, gaussFit(xhelp, *popt), 'k-.', label="fitpoisson")
    plt.legend()

    plt.title(name)
    plt.show()

    return [popt, perr]

data = loadData("40")
#print(data[1], data[2])
chsToEnergy(data)
energies = (data[1], data[2])
ens1 = []
ens2 = []
n = len(data[0])
for i in np.linspace(0, n-1, n):
    #print(i)
    i = int(i)

    if checkConservation(data[1][i], data[2][i], 40):
        ens1 += [data[1][i]]
        ens2 += [data[2][i]]

e1 = np.unique(ens1, return_counts= True)
lI = np.where(e1[0]>0)
#print(lI)
plt.plot(e1[0], e1[1])
e2 = np.unique(ens2, return_counts= True)
lI = np.where(e1[0]>0)
plt.title("1")
plt.show()
#print(lI)
plt.plot(e2[0], e2[1])
plt.title("2")
plt.show()

test2 = getFit("test", e1, [661, 100, 100])
getFit("test2", e2, [500, 100, 100])
#print(test2)

for i in [40, 60, 80, 100, 116]:
    print(i, conservation(600, i))

# incidence = np.loadtxt(folder + "/compton test 1_ch000.txt", skiprows=4)
# output = np.loadtxt(folder + "/compton test 1_ch001.txt", skiprows=4)
# toDelete = []
# for i in np.linspace(0, len(incidence)-1, len(incidence)):
#     i = int(i)
#     if np.abs(incidence[i,1]-output[i,1]) > 1000:
#         toDelete = toDelete + [i]
# data = np.array([incidence[:,0], incidence[:,1], output[:,1]])
# data = np.array([np.delete(incidence[:,0], toDelete), np.delete(incidence[:,1], toDelete), np.delete(output[:,1], toDelete)])
# diffs = data[1]-data[2]
# (x, y) = np.unique(data[2], return_counts=True)
# lI = np.where(x == 1)[0][0]
# hI = np.where(x > 1000)[0][0]
# x = x[lI:hI]
# y = y[lI:hI]
# plt.plot(x, y)
# plt.show()
# print(incidence[0,1])
#
# getChannel("test", (x,y), 510, 700, [600, 1, 1])

