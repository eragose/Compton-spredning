import numpy as np
import matplotlib.pyplot as plt

timefilteredBGO = np.loadtxt('Co/Compton_spredning 40_ch000_timefiltered.txt')
timefilteredNaI = np.loadtxt('Co/Compton_spredning 40_ch001_timefiltered.txt')
energyfilteredBGO = np.loadtxt('Co/Energy_filtered Compton_spredning 40_ch000.txt')
energyfilteredNaI = np.loadtxt('Co/Energy_filtered Compton_spredning 40_ch001.txt')
impulsfilteredBGO = np.loadtxt('Co/Impuls_filtered Compton_spredning 40_ch000.txt')
impulsfilteredNaI = np.loadtxt('Co/Impuls_filtered Compton_spredning 40_ch001.txt')
plt.scatter(timefilteredBGO[:, 1], timefilteredNaI[:, 1], alpha=0.05)
plt.scatter(energyfilteredBGO[:, 1], energyfilteredNaI[:, 1], alpha=0.05)
plt.scatter(impulsfilteredBGO[:, 1], impulsfilteredNaI[:, 1], alpha=0.05)
plt.show()


