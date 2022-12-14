import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
folder = 'Co/'
seperator = ' '
file_type = '.txt'


def loadData(name):
    dat0 = np.loadtxt(folder + "Impuls_filtered Compton_spredning " + name + "_ch000" + file_type)
    dat1 = np.loadtxt(folder + "Impuls_filtered Compton_spredning " + name + "_ch001" + file_type)
    return dat0, dat1


def gaussFit(x, mu, sig, a, b, c):
    lny = np.log(a) - ((x-mu)**2)/(2*sig**2)
    return np.exp(lny)+b*x+c


def chToEnergy(ch):
    return 1.12166*ch-18.19


def getFit(name: str, data: tuple, guess: [int, int, int], lower_limit: int = 0, upper_limit: int = 700, fun = gaussFit, guess2= [0,0], plot=False):
    # print(np.where(data[0]>lower_limit))
    ll = np.where(data[0]>lower_limit)[0][0]
    ul = np.where(data[0]<upper_limit)[0][-1]
    x = data[0][ll:ul]
    y = data[1][ll:ul]
    yler = y
    pinit = guess + guess2
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

    if plot:
        plt.xlabel('Channel')
        plt.ylabel('Counts')
        plt.scatter(x, y, color="r", label="data")
        plt.plot(xhelp, fun(xhelp, *popt), 'k-.', label="fitgauss")
        plt.legend()
        plt.title(name + ' gaussfit')
        plt.savefig(name + ' gaussfit.pdf')
        plt.show()

    return [popt, perr]


def checkcompton(data, angle, plot=False):
    name = str(angle)
    NaIChannels, NaICounts = np.unique(data[1][:, 1], return_counts=True)
    #NaICounts, NaIChannels = np.histogram(data[1][:, 1], bins='auto')
    #for i in range(np.size(NaIChannels)-1):
    #    i = int(i)
    #    NaIChannels[i] = (NaIChannels[i]+NaIChannels[i+1])/2
    expected = conservation(angle)
    NaI = getFit('NaI ' + name, (NaIChannels, NaICounts), [expected, 10, 10], plot=plot)
    BGO = getFit('BGO ' + name, np.unique(data[0][:, 1], return_counts=True), [661-expected, 10, 10], plot=plot)
    plt.xlabel('BGO')
    plt.ylabel('NaI')
    plt.title('BGO vs NaI ' + name + ' + markering af compton zone')
    BGOenergy = BGO[0][0]
    NaIenergy = NaI[0][0]
    BGOuncertainty = BGO[0][1]
    NaIuncertainty = NaI[0][1]
    plt.axvline(BGOenergy, label='measured energy')
    plt.axhline(NaIenergy)
    plt.axvspan(BGOenergy - BGOuncertainty, BGOenergy + BGOuncertainty, alpha=0.2,
                label='Energy confidence interval')
    plt.axhspan(NaIenergy - NaIuncertainty, NaIenergy + NaIuncertainty, alpha=0.2)
    plt.scatter(data[0][:, 1], data[1][:, 1], alpha=0.1, label='Datapoints')
    plt.legend()
    plt.show()
    return NaI


def conservation( theta):
    mc2 = 0.5*1000 #keV
    E1 = 661.7
    return E1/(1+(E1/mc2)*(1-np.cos(theta*np.pi/180)))


dat40 = loadData('40')
NaI40 = checkcompton(dat40, 40)

dat60 = loadData('60')
NaI60 = checkcompton(dat60, 60)

dat80 = loadData('80')
NaI80 = checkcompton(dat80, 80)

dat100 = loadData('100')
NaI100 = checkcompton(dat100, 100)

dat116 = loadData('116')
NaI116 = checkcompton(dat116, 116)

print('test80', conservation(80))

# Plots the output energies
oEs = [NaI40[0][0], NaI60[0][0], NaI80[0][0], NaI100[0][0], NaI116[0][0]]
oEes = [[NaI40[1][0], NaI60[1][0], NaI80[1][0], NaI100[1][0], NaI116[1][0]]]

angles = np.array([40, 60, 80, 100, 116])
angleErr = angles**0

plt.errorbar(angles, oEs, yerr=oEes, xerr=angleErr, fmt=",", label='measured energies')
angleHelp = np.linspace(angles[0]-5, angles[-1]+4, 100)
plt.plot(angleHelp, conservation(angleHelp), label='Theoretical energy')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Energy (KeV)')
plt.title('Comparison with theoretical energy')
plt.legend()
plt.savefig('Comparison with theoretical energy.pdf')
plt.show()

NaIparams = [NaI40, NaI60, NaI80, NaI100, NaI116]

times = np.loadtxt('times')
Is = []
Ies = []
for i in range(len(NaIparams)):
    i = int(i)
    #print(params[i][1][0][0])
    sig = NaIparams[i][0][1]
    amp = NaIparams[i][0][2]
    area = sig * amp * 2 * np.pi
    sigerr = NaIparams[i][1][1]
    amperr = NaIparams[i][1][2]
    areaUncertaintyA = np.sqrt((sigerr / sig)**2 + (amperr / amp)**2) * 2 * np.pi
    areaUncertainty = areaUncertaintyA * area
    Is += [area/times[i,1]]
    Ies += [areaUncertainty/times[i,1]]

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
def probfit(theta, a):
    return a*prob(theta)

plt.errorbar(angles, Is, yerr=Ies, xerr=angleErr, fmt=".")
plt.xlabel('Angle (degrees)')
plt.ylabel('Intensity (counts/s)')
plt.title('Measured intensity')
plt.savefig('Measured intensity.pdf')
#plt.show()
popt, pcov = curve_fit(probfit, angles, Is, 26, Ies, absolute_sigma=True)
angleHelp = np.linspace(35, 120, 180)
plt.plot(angleHelp, probfit(angleHelp, *popt))
plt.xlabel('Angle (degrees)')
plt.ylabel('Probability')
plt.title('Theoretical intensity')
plt.savefig('Comparison with theoretical intensity.pdf')
plt.show()

dats = [dat40, dat60, dat80, dat100, dat116]
Is=[]
for i in range(len(dats)):
    i = int(i)
    I = np.sum(dats[i][1][:,1])
    t = times[i][1]
    Is += [I/t]

plt.scatter(angles, Is)
plt.show()



