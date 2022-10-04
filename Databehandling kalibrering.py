import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

folder = 'kalibrering'
seperator = ' '
file_type = '*.txt'



read_files = glob.glob(os.path.join(folder, file_type))

#Creates a list of all file names
np_array_values = []
for files in read_files:
    pdfile = pd.read_csv(files, sep=seperator, skiprows=3)           #Specify seperator
    np_array_values.append(pdfile)

print(np_array_values[0])

def getCounts(name: str, lc: int = 1, hc: int = 2000):
    data = np.loadtxt("kalibrering/Kali" + name + "_ch001.txt", skiprows=4)
    counts = data[:, 1]
    (x, y) = np.unique(counts, return_counts=True)
    lI = np.where(x == lc)[0][0]
    hI = np.where(x == hc)[0][0]
    x = x[lI:hI]
    y = y[lI:hI]
    plt.plot(x, y)
    plt.title(name)
    plt.show()
    return (x, y)

def gaussFit(x, mu, sig, a, b, c):
    lny = np.log(a) - ((x-mu)**2)/(2*sig)
    return np.exp(lny) - (b*x+c)
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

    return popt


#Am = getCounts("Am", hc = 2100)
Cs = getCounts("Caeseium")
Co = getCounts("Co")
#Na = getCounts("Na")
Ra = getCounts("Ra")

CsCh = getChannel("Cs channel", Cs, 500, 750, [600, 10, 1000])
RaCh = getChannel("Ra channel", Ra, 500, 700, [550, 10, 500])
CoCh = getChannel("Co channel", Co, 1000, 1150, [1050, 10, 10])
x = [CsCh[0], RaCh[0], CoCh[0]]
y = [661.661, 609, 1173.238]
yler = [0.03, 1, 0.015]

def funlin(x, a, b):
    return a*x+b
yler = np.sqrt(y)
pinit = [1,1]
xhelp = np.linspace(0, 2000, 500)
popt, pcov = curve_fit(funlin, x, y, p0=pinit, sigma=yler, absolute_sigma=True)

print('a hældning:', popt[0])
print('b forskydning:', popt[1])

perr = np.sqrt(np.diag(pcov))
print('usikkerheder:', perr)
#chmin = np.sum(((y - funlin(x, *popt)) / yler) ** 2)
#print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4))
plt.scatter(x, y, label="data")
plt.plot(xhelp, funlin(xhelp, *popt), label="fit")
plt.legend()
plt.show()





