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

def getCounts(name: str):
    data = np.loadtxt("kalibrering/Kali" + name + "_ch001.txt", skiprows=4)
    counts = data[:, 1]
    (x, y) = np.unique(counts, return_counts=True)
    lI = np.where(x == 1)[0][0]
    hI = np.where(x == 2000)[0][0]
    x = x[lI:hI]
    y = y[lI:hI]
    plt.plot(x, y)
    plt.title(name)
    plt.show()
    return (x, y)

(xAm, yAm) = getCounts("Am")
(xCs, yCs) = getCounts("Caeseium")
(xCo, yCo) = getCounts("Co")
(xNa, yNa) = getCounts("Na")
(xRa, yRa) = getCounts("Ra")





