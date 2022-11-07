import numpy as np
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


def conservationtest(E1, E2, theta):
    mc2 = 0.5*1000 #keV
    Etot = E1+E2
    return Etot/(1+(Etot/mc2)*(1-np.cos(theta*np.pi/180)))


def conservation(theta):
    mc2 = 0.5*1000 #keV
    E1 = 661
    return E1/(1+(E1/mc2)*(1-np.cos(theta*np.pi/180)))


angles = [40, 60, 80, 100, 116]
for i in angles:
    theta = i
    i = str(i)
    dat0, dat1 = loadData(i)

    #toDelete = np.where(np.abs(conservationtest(dat0[:, 1], dat1[:, 1], theta)-dat1[:, 1]) > 50)
    toDelete = np.where(np.abs(conservation(theta) - dat1[:, 1]) > 60)
    #print(np.where(dat1[:,1]<0))
    #rint(dat0[713])
    #print(dat1[713])

    plt.scatter(dat0[:, 1], dat1[:, 1], alpha=0.05)

    dat0 = np.delete(dat0, toDelete, 0)
    dat1 = np.delete(dat1, toDelete, 0)
    #dats += [(dat0, dat1)]

    plt.scatter(dat0[:, 1], dat1[:, 1], alpha=0.01, color='r')

    plt.title("BGO vs NaI " + str(i) + " degrees")
    plt.xlabel("BGO")
    plt.ylabel("NaI")
    plt.show()
    np.savetxt(folder + 'Impuls_filtered Compton_spredning ' + i + '_ch000.txt', dat0, delimiter=" ", fmt='%s')
    np.savetxt(folder + 'Impuls_filtered Compton_spredning ' + i + '_ch001.txt', dat1, delimiter=" ", fmt='%s')
