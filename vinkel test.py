import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
folder = 'Data angles'
seperator = ' '
file_type = '*.txt'

def gaussFit(x, mu, sig, a, b, c):
    lny = np.log(a) - ((x-mu)**2)/(2*sig)
    return np.exp(lny) - (b*x+c)

def loadData(name):
    (time, ch0) = np.loadtxt(folder + "compton test" + name + "ch000" + file_type)
    (time, ch1) = np.loadtxt(folder + "compton test" + name + "ch001" + file_type)
    return (time, ch0, ch1)

def ChToEnergy(ch):
    return 1.12166*ch-18.19

def ChsToEnergy(data):
    size = len(data[0])
    for i in np.linspace(0, size-1, size):
        i = int(i)
        data[1][i] = ChToEnergy(data[1][i])
        data[2][i] = ChToEnergy(data[2][i])


def checkConservation(E1, E2, theta):
    {}
def getChannel(name: str, data: tuple, lower_limit: int, upper_limit: int, guess: [int, int, int]):
    x = data[0][lower_limit:upper_limit]
    y = data[1][lower_limit:upper_limit]
    yler = np.sqrt(y)
    pinit = guess + [0,0]
    xhelp = np.linspace(lower_limit, upper_limit, 500)
    popt, pcov = curve_fit(gaussFit, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
    print(name)
    print('mu :', popt[0])
    print('sigma :', popt[1])
    print('scaling', popt[2])
    print('background', popt[3], popt[4])
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

incidence = np.loadtxt(folder + "/compton test 1_ch000.txt", skiprows=4)
output = np.loadtxt(folder + "/compton test 1_ch001.txt", skiprows=4)
toDelete = []
for i in np.linspace(0, len(incidence)-1, len(incidence)):
    i = int(i)
    if np.abs(incidence[i,1]-output[i,1]) > 1000:
        toDelete = toDelete + [i]
data = np.array([incidence[:,0], incidence[:,1], output[:,1]])
data = np.array([np.delete(incidence[:,0], toDelete), np.delete(incidence[:,1], toDelete), np.delete(output[:,1], toDelete)])
diffs = data[1]-data[2]
(x, y) = np.unique(data[2], return_counts=True)
lI = np.where(x == 1)[0][0]
hI = np.where(x > 1000)[0][0]
x = x[lI:hI]
y = y[lI:hI]
plt.plot(x, y)
plt.show()
print(incidence[0,1])

getChannel("test", (x,y), 510, 700, [600, 1, 1])

