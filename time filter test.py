import numpy as np
import matplotlib.pyplot as plt
folder = 'Co/'
seperator = ' '
file_type = '.txt'


def loadData(name):
    dat0 = np.loadtxt(folder + "Comptorn spredning " + name + "_ch000" + file_type, skiprows=4)
    dat1 = np.loadtxt(folder + "Comptorn spredning " + name + "_ch001" + file_type, skiprows=4)
    return dat0, dat1

dats=[]
timedifferences = np.array([])
timedifferences1 = np.array([])
angles = [40, 60, 80, 100, 116]
times = []
for i in angles:
    i = str(i)
    dat0, dat1 = loadData(i)

    toDelete = np.where(dat0[:, 1] < 1)
    #print(np.where(dat1[:,1]<0))
    #rint(dat0[713])
    #print(dat1[713])
    toDelete = np.append(toDelete, np.where(dat1[:, 1] < 1))
    toDelete = np.append(toDelete, np.where(dat1[:, 1] > 800))
    toDelete = np.append(toDelete, np.where(dat0[:, 1] > 800))
    times += [[i, (dat1[-1, 0]-dat0[0, 0])*10**(-8)]]
    timedifferences1 = dat1[:, 0] - dat0[:, 0]
    toDelete = np.append(toDelete, np.where(timedifferences1 > 100))
    dat0 = np.delete(dat0, toDelete, 0)
    dat1 = np.delete(dat1, toDelete, 0)
    dats += [(dat0, dat1)]
    #timedifferences = dat1[:, 0]-dat0[:, 0]
    plt.scatter(dat0[:, 1], dat1[:, 1], alpha=0.05)
    plt.title("BGO vs NaI " + str(i) + " degrees")
    plt.xlabel("BGO")
    plt.ylabel("NaI")
    plt.show()
    np.savetxt(folder + 'Compton_spredning ' + i + '_ch000_timefiltered.txt', dat0, delimiter=" ", fmt='%s')
    np.savetxt(folder + 'Compton_spredning ' + i + '_ch001_timefiltered.txt', dat1, delimiter=" ", fmt='%s')
    np.savetxt('Times', times, delimiter=" ", fmt='%s')

