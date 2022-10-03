import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



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