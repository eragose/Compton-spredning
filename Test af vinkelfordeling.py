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


def gaussFit(x, mu, sig, a, b, c):
    lny = np.log(a) - ((x-mu)**2)/(2*sig**2)
    return np.exp(lny)+b*x+c


def getFit(name: str, data: tuple, guess: [int, int, int], lower_limit: int = 0, upper_limit: int = 1000, fun = gaussFit, guess2= [0,0]):
    # print(np.where(data[0]>lower_limit))
    ll = np.where(data[0]>lower_limit)[0][0]
    ul = np.where(data[0]<upper_limit)[0][-1]
    x = data[0][ll:ul]
    y = data[1][ll:ul]
    yler = y
    pinit = guess + guess2
    xhelp = np.linspace(lower_limit, upper_limit, 500)
    popt, pcov = curve_fit(fun, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
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
    chmin = np.sum(((y - fun(x, *popt)) / yler) ** 2)
    print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4))
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    #plt.text(0.05, 1.05, text, fontsize=14,
    #        verticalalignment='top', bbox=props)
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.scatter(x, y, color="r", label="data")
    plt.plot(xhelp, fun(xhelp, *popt), 'k-.', label="fitgauss")
    plt.legend()

    plt.title(name)
    plt.show()

    return [popt, perr]


dat0, dat1 = loadData('40')
NaIChannels, NaICounts = np.unique(dat1[:,1], return_counts=True)
NaI40 = getFit('NaI 40', (NaIChannels, NaICounts), [500,10,10])
BGO40 = getFit('BGO 40', np.unique(dat0[:,1], return_counts=True), [100,10,10])

plt.xlabel('BGO')
plt.ylabel('NaI')
plt.title('BGO vs Nai 40 + markering af compton zone')
BGO40energy = BGO40[0][0]
NaI40energy = NaI40[0][0]
BGO40uncertainty = BGO40[0][1]
NaI40uncertainty = NaI40[0][1]
plt.axvline(BGO40energy, label='measuered energy')
plt.axhline(NaI40energy)
plt.axvspan(BGO40energy-BGO40uncertainty, BGO40energy+BGO40uncertainty, alpha=0.2, label='Energy confidence interval')
plt.axhspan(NaI40energy-NaI40uncertainty, NaI40energy+NaI40uncertainty, alpha=0.2)
plt.scatter(dat0[:,1], dat1[:, 1], alpha=0.1, label='Datapoints')
plt.legend()
plt.show()