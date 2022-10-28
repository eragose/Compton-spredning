from re import A
#from turtle import back
import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import zip_longest

folder = 'Attenuation'
seperator = ' '
file_type = '*.txt'




read_files = glob.glob(os.path.join(folder, file_type))

#Creates a list of all file names
np_array_values = []
for files in read_files:
    pdfile = pd.read_csv(files, sep=seperator, skiprows=3)           #Specify seperator
    np_array_values.append(pdfile)

print(np_array_values)
def getData(name):
    return np.loadtxt("Attenuation/Attenuation Aluminium " + name + "_ch001.txt", skiprows=4)

def getTime(data):
    times = data[:,0]
    start = times[0]
    end = times[-1]
    return (end-start)/10**8
def getCounts(data, name: str, lc: int = 1, hc: int = 1000):
    counts = data[:, 1]
    (x, y) = np.unique(counts, return_counts=True)
    lI = np.where(x == lc)[0][0]
    hI = np.where(x >= hc)[0][0]
    x = x[lI:hI]
    y = y[lI:hI]
    #plt.plot(x, y)
    #plt.title(name)
    #plt.show()
    return (x, y)

def gaussFit(x, mu, sig, a, b, c):
    lny = np.log(a) - ((x-mu)**2)/(2*sig)
    return np.exp(lny) - (b*x+c)

def getChannel(name: str, data: tuple, lower_limit: int, upper_limit: int, guess: [int, int, int]):
    ll = np.where(data[0] > lower_limit)[0][0]
    ul = np.where(data[0] < upper_limit)[0][-1]
    x = data[0][ll:ul]
    y = data[1][ll:ul]
    yler = np.sqrt(y)
    pinit = guess + [30,-5]
    print(pinit)
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
    plt.plot(x, y, label="data")
    plt.plot(xhelp, gaussFit(xhelp, *popt), label="fit")
    plt.legend()
    plt.title(name)
    plt.show()



    return [popt, perr]

intensities = []
for i in np.linspace(0,7,8):
    i = str(int(i))
    data = getData(i)
    time = getTime(data)
    countData = getCounts(data, i)
    fit = getChannel(i, countData, 500, 700, [640, 10, 10])
    # area = sigma * amplitude * 2 pi
    area = fit[0][1]*fit[0][2]*2*np.pi
    # uncertainty = swqrt(sigmaerr/sigma+amplitudeerr/amplitude)*2*pi
    areaUncertaintyA = np.sqrt((fit[1][1]/fit[0][1])+(fit[1][2]/fit[0][2]))*2*np.pi
    areaUncertainty = areaUncertaintyA/area
    intensity = area/time
    intensityUncertainty = areaUncertainty#/time
    intensities += [[intensity, intensityUncertainty]]
intensities = np.array(intensities)
#Back = getCounts("bg")
#A1 = getCounts("1")
#A2 = getCounts("2")
#A3 = getCounts("3")
#A4 = getCounts("4")
#A5 = getCounts("5")
#A6 = getCounts("6")
#A7 = getCounts("7")
#print(len(Back), len(A1))
def dif(a,b):
    Newx0 = []
    Newx1 = []
    i = 0
    for x1 in a[0]:
        x1 = int(x1)
        j = 0
        for x2 in b[0]:
            x2 = int(x2)
            #print("i: ", i, "j: ", j, "\n")
            if x1 == x2:
                Newx0.append(x1)
                Newx1.append(b[1][j]-a[1][i])
            j += 1
        i += 1
    return Newx0,Newx1



def logFit(x, a, b, c, d):
    return a/np.log(b*x+c)+d

def expFit(x, a, b):
    return a*np.exp(x*b)

x = np.linspace(0,7,8)*10.09
xler = np.append([0], np.linspace(1,7,7)*0.02)
y = intensities[:,0]
yler = intensities[:,1]
pinit = [850, -0.05]
xhelp = np.linspace(0, 100, 100)
popt, pcov = curve_fit(expFit, x, y, p0=pinit, sigma=yler, absolute_sigma= True)
print("intensity fit")
print('a :', popt[0])
print('b :', popt[1])
#print('c :', popt[2])

perr = np.sqrt(np.diag(pcov))
print('usikkerheder:', perr)
chmin = np.sum(((y - expFit(x, *popt)) / yler) ** 2)
print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4))

plt.plot(x, expFit(x, *popt), label = "fit")

plt.errorbar(x, y, yerr = yler, xerr= xler, fmt='o', label = "Intensity")
plt.title('Intensity')
plt.legend()
plt.xlabel("Thickness of aluminium (mm)")
plt.ylabel("Intensity (counts/s")
plt.savefig('Attenuation.pdf')
plt.show()
