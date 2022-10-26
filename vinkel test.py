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
def checkConservation(E1, E2, theta, error1 = 0.1, error2 = 120):
    if E1>0:
        if (np.abs(1-E2/conservation(E1, theta)) < error1):
            if np.abs(E1-E2)>error2:
                return True
            else :
                return False
        else:
            return False
    else:
        return False


def getFit(name: str, data: tuple, guess: [int, int, int], lower_limit: int = 0, upper_limit: int = 1000):
    ll = np.where(data[0]>lower_limit)[0][0]
    ul = np.where(data[0]<upper_limit)[0][-1]
    x = data[0][ll:ul]
    y = data[1][ll:ul]
    yler = y
    pinit = guess
    xhelp = np.linspace(lower_limit, upper_limit, 500)
    popt, pcov = curve_fit(gaussFit, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    text = ""
    text += "       mu          sigma           scaling " + "\n"
    text += "values" + str(popt) + "\n"
    text += "errors" + str(perr)
    print(name)
    print(text)
    #print('mu :', popt[0])
    #print('sigma :', popt[1])
    #print('scaling', popt[2])
    #print('background', popt[3], popt[4])

    #print('usikkerheder:', perr)
    chmin = np.sum(((y - gaussFit(x, *popt)) / yler) ** 2)
    print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4))
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    #plt.text(0.05, 1.05, text, fontsize=14,
    #        verticalalignment='top', bbox=props)
    plt.plot(x, y, color="r", label="data")
    plt.plot(xhelp, gaussFit(xhelp, *popt), 'k-.', label="fitpoisson")
    plt.legend()

    plt.title(name)
    plt.show()

    return [popt, perr]
def filter(theta, data, error1 = 0.11, error2 = 120):
    chsToEnergy(data)
    energies = (data[1], data[2])
    ens1 = []
    ens2 = []
    n = len(data[0])
    for i in np.linspace(0, n-1, n):
        i = int(i)
        if checkConservation(data[1][i], data[2][i], theta, error1, error2):
            ens1 += [data[1][i]]
            ens2 += [data[2][i]]
    e1 = np.unique(ens1, return_counts= True)
    e2 = np.unique(ens2, return_counts= True)
    return (e1, e2)
angles = [40, 60, 80, 100, 116]

energies = []
params = []
times = []
for i in angles:
    stri = str(i)
    data = loadData(stri)
    t1 = data[0][0]
    t2 = data[0][-1]
    time = (t2-t1)/10**8
    (e1, e2) = filter(i, data, error1=0.1, error2=600-conservation(600, i))
    energies += [[i, e1, e2]]
    expectedE2 = conservation(600, i)
    E1 = getFit(stri + ": E1", e1, [660, 1000, 1000])
    E2 = getFit(stri + ": E2", e2, [expectedE2, 1000, 1000])
    params += [[E1, E2]]
    times += [time]

oEs = []
oEes = []
pRange = range(len(params))
for i in pRange:
    #print(params[i][1][0][0])
    oEs += [params[i][1][0][0]]
    oEes += [params[i][1][1][0]]


def printCons(angle, energy):
    print(angle, conservation(energy, angle))
printCons(116, 200)


