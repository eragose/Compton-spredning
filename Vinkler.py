import glob
import os
import csv

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

<<<<<<< HEAD
folder = 'Co'
seperator = ' '
file_type = '.dat'

def BinaryToDecimal(binary):
        
    binary1 = binary
    decimal, i, n = 0, 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
    return (decimal)  

#df = pd.read_csv('Comptorn spredning 40_ch000.dat', sep='\s+|\s+')
#df.to_csv('Comptorn spredning 40_ch000.dat', index=None)


# with open('Comptorn spredning 40_ch000.dat', 'r') as dat_file:
#     with open('Comptorn spredning 40_ch000.csv', 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         for row in dat_file:
# #             row = [value.strip() for value in row.split('|')]
# #             csv_writer.writerow(row)

# with open("Comptorn spredning 60_ch000.dat",'rb',) as datFile:
#     #print([data.split() for data in datFile])
#     X = []
#     x0 = []
    
#     for i in datFile: 
#         #print(i)
#         x = int.from_bytes(i,'little')
#         X.append(x)
#         x0.append(i)

#     for _ in
#     print(x0[0:7])   





file = open("Comptorn spredning 40_ch000.dat", "rb")

# Read the first five numbers into a list

number = int(file.read(1),2)

# Print the list

print(number)

# Close the file

file.close()
=======
def get_time_counts(angle):
    angle = str(angle)
    data_0 = np.loadtxt("Co\Comptorn spredning "+ angle + "_ch000.txt", skiprows=4)
    data_1 = np.loadtxt("Co\Comptorn spredning "+ angle + "_ch001.txt", skiprows=4)
    #data fra 000
    time_in_02us = data_0[:,0]
    count_channel_0 = data_0[:,1]
    #data fra 001
    count_channel_1 = data_1[:,1]

    time = (time_in_02us[-1] - time_in_02us[0])*(10**(-8))
    print((time_in_02us[-1]- time_in_02us[0])*(10**(-8)))
    counts_0 = np.unique(count_channel_0[count_channel_0 > -100], return_counts=True)
    counts_1 = np.unique(count_channel_1[count_channel_1 > -100], return_counts=True)

    
    return time, counts_0[0], counts_0[1], counts_1[0], counts_1[1]

def energy_fuc(x):
    a = 1.1216620315946357
    b = -18.18580680037887
    return a*x+b

angles = [40, 60, 80, 100, 116]
deg40 = get_time_counts(angles[0])
plt.scatter(deg40[1], deg40[2])
plt.show()


>>>>>>> 6e7dde2e1fed9e9381fe5304b378a6691cb16486
