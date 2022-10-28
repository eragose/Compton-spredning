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


def chToEnergy(ch):
    return 1.12166*ch-18.19


def chsToEnergy(data):
    size = len(data[0])
    for i in np.linspace(0, size-1, size):
        i = int(i)
        data[1][i] = chToEnergy(data[1][i])
        data[2][i] = chToEnergy(data[2][i])
    return data


def conservation(E1, theta):
    mc2 = 0.5*1000 # keV
    return E1/(1+(E1/mc2)*(1-np.cos(theta*np.pi/180)))


def gaussFit(x, mu, sig, a):
    lny = np.log(a) - ((x-mu)**2)/(2*sig)
    return np.exp(lny)+1


def getFit(name: str, data, guess: [int, int, int], lower_limit: int = 0, upper_limit: int = 1000, fun = gaussFit):
    # print(np.where(data[0]>lower_limit))
    ll = np.where(data[0]>lower_limit)[0][0]
    ul = np.where(data[0]<upper_limit)[0][-1]
    x = data[0][ll:ul]
    y = data[1][ll:ul]
    yler = y
    pinit = guess
    xhelp = np.linspace(lower_limit, upper_limit, 500)
    popt, pcov = curve_fit(fun, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    text = ""
    text += "       mu          sigma           scaling " + "\n"
    text += "values" + str(popt) + "\n"
    text += "errors" + str(perr)
    print(name)
    print(text)

    chmin = np.sum(((y - fun(x, *popt)) / yler) ** 2)
    print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4))
    plt.scatter(x, y, color="r", label="data")
    plt.plot(xhelp, fun(xhelp, *popt), 'k-.', label="fitgauss")
    plt.legend()

    plt.title(name)
    plt.show()

    return [popt, perr]



dats = []
angles = [40, 60, 80, 100]
for i in angles:
    i = str(i)
    dats += [loadData(i)]

counts = []
counts1 = []
times = []
for dat in dats:
    dat = chsToEnergy(dat)
    counts += [np.unique(dat[2], return_counts=True)]
    counts1 += [np.unique(dat[1], return_counts=True)]
    times += (dat[0][-1]-dat[0][0])/10**(-8)

plt.plot(counts[0][0], counts[0][1])
plt.show()
plt.plot(counts[2][0], counts[2][1])
plt.plot(counts1[2][0], counts1[2][1])
plt.show()

for i in range(len(angles)):
    expectation = conservation(661, angles[i])
    getFit(str(angles[i]), counts[i], [expectation, 200, 20], 0, 1000)

conservation(661, 80)


print(conservation(661, 40))





