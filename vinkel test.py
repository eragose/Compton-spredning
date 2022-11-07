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


def getFit(name: str, data: tuple, guess: [int, int, int], lower_limit: int = 0, upper_limit: int = 1000, fun = gaussFit):
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
    plt.scatter(x, y, color="r", label="data")
    plt.plot(xhelp, fun(xhelp, *popt), 'k-.', label="fitgauss")
    plt.legend()

    plt.title(name)
    plt.show()

    return [popt, perr]
def filter(theta, data, error1 = 0.11, error2 = 120, binsize=0):
    chsToEnergy(data)

    ens1 = []
    ens2 = []
    n = len(data[0])
    for i in np.linspace(0, n-1, n):
        i = int(i)
        if checkConservation(data[1][i], data[2][i], theta, error1, error2):
            ens1 += [data[1][i]]
            ens2 += [data[2][i]]
    #e2 = np.histogram(ens2)
    #means = np.array([])
    #for i in range(e2hist[1].size-1):
    #    n = e2hist[1][i]+e2hist[1][i+1]/2
    #    print(n)
    #    np.append(means, n)
    #e2 = (means, e2hist[0])
    e1 = np.unique(ens1, return_counts= True)
    e2 = np.unique(ens2, return_counts= True)
    #print(e2)
    #print(e2[0].size/2)
    def dobin(binsize, e):
        while e[0].size % binsize > 0:
            e = (np.delete(e[0], -1), np.delete(e[1], -1))
        e20 = np.array([])
        e21 = np.array([])
        for i in np.linspace(0, e[0].size-binsize, int(e[0].size/binsize)):
            i = int(i)
            #print(i)
            add=0
            avg=0
            for j in range(binsize):
                #print(j)
                add += e[1][i + j]
                avg += e[0][i + j]
            avg = avg/binsize
            e20 = np.append(e20, avg)
            e21 = np.append(e21, add)
        return (e20, e21)
    if binsize > 1:
        e2 = dobin(binsize, e2)
    #print(e2)

    return (e1, e2)







angles = [40, 60, 80, 100, 116]
angleErr = np.array(angles)**0*0.5

# gets all input output peak fit parameters and plots them on top of data.
# Also returns the collection time in "times"

energies = []
params = []
times = []
pRange = np.linspace(0, len(angles)-1, len(angles))
sizes = [13,9,4,3,2]


def getandfit(angle, binsize):

    stri = str(angle)
    i = angle
    print(i)
    data = loadData(stri)
    t1 = data[0][0]
    t2 = data[0][-1]
    time = (t2-t1)/10**8
    (e1, e2) = filter(i, data, error1=0.1, error2=600-conservation(600, i), binsize=binsize)
    #print(e1[1][0], type(e1[1][0]))
    # e1 = [e1[0], np.histogram(e1[1], bins=int(e1[1].size/2))]
    # e2 = [e2[0], np.histogram(e2[1], bins=int(e2[1].size/2))]
    def binnedGauss(x, mu, sig, a):
        lny = np.log(a) - ((x - mu) ** 2) / (2 * sig)
        return np.exp(lny) + binsize
    expectedE2 = conservation(600, i)
    E1 = getFit(stri + ": E1", e1, [660, 1000, 1000])
    E2 = getFit(stri + ": E2", e2, [expectedE2, 1000, 1000], fun = binnedGauss)
    return e1, e2, E1, E2, time


# getandfit(40, 3)
for i in pRange:
    i = int(i)
    print(i)
    theta = angles[i]
    binsize = sizes[i]
    e1, e2, E1, E2, time = getandfit(theta, binsize)
    energies += [[e1, e2]]
    params += [[E1, E2]]
    times += [time]

# Plots the output energies
oEs = []
oEes = []

for i in pRange:
    i = int(i)
    #print(params[i][1][0][0])
    oEs += [params[i][1][0][0]]
    oEes += [params[i][1][1][0]]

plt.errorbar(angles, oEs, yerr=oEes, xerr=angleErr, fmt=",")
angleHelp = np.linspace(angles[0], angles[-1], 100)
plt.plot(angleHelp, conservation(661.661, angleHelp))
plt.show()

iEs = []
iEes = []

for i in pRange:
    i = int(i)
    #print(params[i][1][0][0])
    iEs += [params[i][0][0][0]]
    iEes += [params[i][0][1][0]]


def const(x, a):
    return x*0+a


plt.errorbar(pRange, iEs, yerr=iEes, fmt='.')
xs = np.array(iEs)
alphas = np.array(iEes)
average = np.sum(xs/alphas**2)/np.sum(1/alphas**2)

popt, pcov = curve_fit(const, np.array(pRange), iEs, p0=660, sigma=iEes, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
print('Energy :', popt[0])
plt.plot(pRange, const(np.array(pRange), popt[0]))
plt.show()


events = []
for i in pRange:
    i = int(i)
    events += [np.sum(energies[i][1][1])/times[i]]

plt.scatter(angles, events)
plt.show()


Is = []
Ies = []
for i in pRange:
    i = int(i)
    #print(params[i][1][0][0])
    sig = params[i][1][0][1]
    amp = params[i][1][0][2]
    area = sig * amp * 2 * np.pi
    sigerr = params[i][1][1][1]
    amperr = params[i][1][1][2]
    areaUncertaintyA = np.sqrt((sigerr / sig) + (amperr / amp)) * 2 * np.pi
    areaUncertainty = areaUncertaintyA / area
    Is += [area/times[i]]
    Ies += [areaUncertainty]

def prob(theta):
    alpha = 661.661  # keV
    r0 = 28.18  # fm
    theta = theta*np.pi/180
    brack1 = (1/(1+alpha*(1-np.cos(theta))))**3
    brack2 = (1+np.cos(theta))/2
    brack3top = alpha**2*(1-np.cos(theta))**2
    brack3bot = ((1+np.cos(theta)**2)*(1+alpha*(1-np.cos(theta))))
    brack3 = (1+brack3top/brack3bot)
    return (r0**2)*brack1*brack2*brack3
plt.errorbar(angles, Is, yerr=Ies, xerr=angleErr, fmt=".")
angleHelp = np.linspace(0, 180, 180)
plt.show()
plt.plot(angleHelp, prob(angleHelp))
plt.show()

def printCons(angle, energy):
    print(angle, conservation(energy, angle))
printCons(116, 200)


