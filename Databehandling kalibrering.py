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
def getCounts():{

}
print(np_array_values[0])

KaliAm = np.loadtxt("kalibrering/KaliAm_ch001.txt", skiprows = 4)
tAm = KaliAm[:, 0]
countsAm = KaliAm[:, 1]
(xAm, yAm) = np.unique(countsAm, return_counts=True)
lI = np.where(xAm == 0)[0][0]
hI = np.where(xAm == 1000)[0][0]
xAm = xAm[lI:hI]
yAm = yAm[lI:hI]

plt.plot(xAm, yAm)
plt.show()
KaliCs = np.loadtxt("kalibrering/KaliCaeseium_ch001.txt", skiprows = 4)
tCs = KaliCs[:, 0]
countsCs = KaliCs[:, 1]