import numpy as np
import matplotlib.pyplot as plt

timefilteredBGO = np.loadtxt('Co/Compton_spredning 40_ch000_timefiltered.txt')
timefilteredNaI = np.loadtxt('Co/Compton_spredning 40_ch001_timefiltered.txt')
energyfilteredBGO = np.loadtxt('Co/Energy_filtered Compton_spredning 40_ch000.txt')
energyfilteredNaI = np.loadtxt('Co/Energy_filtered Compton_spredning 40_ch001.txt')
impulsfilteredBGO = np.loadtxt('Co/Impuls_filtered Compton_spredning 40_ch000.txt')
impulsfilteredNaI = np.loadtxt('Co/Impuls_filtered Compton_spredning 40_ch001.txt')
plt.scatter(timefilteredBGO[:, 1], timefilteredNaI[:, 1], alpha=0.05,
            label='Filtered for coincidence (Blue)')
plt.scatter(energyfilteredBGO[:, 1], energyfilteredNaI[:, 1], alpha=0.05,
            label='Filtered for conservation of energy (Yellow)')
plt.scatter(impulsfilteredBGO[:, 1], impulsfilteredNaI[:, 1], alpha=0.05,
            label='Filtered for conservation of momentum (Green)')
plt.legend()
plt.title('Events at different levels of filtering')
plt.xlabel('BGO (keV)')
plt.ylabel('NaI (keV)')
plt.savefig('VennDiagram.pdf')
plt.show()


