import glob
import os
import csv

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

folder = 'Co'
seperator = ' '
file_type = '.dat'

#df = pd.read_csv('Comptorn spredning 40_ch000.dat', sep='\s+|\s+')
#df.to_csv('Comptorn spredning 40_ch000.dat', index=None)


# with open('Comptorn spredning 40_ch000.dat', 'r') as dat_file:
#     with open('Comptorn spredning 40_ch000.csv', 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         for row in dat_file:
#             row = [value.strip() for value in row.split('|')]
#             csv_writer.writerow(row)

with open("Comptorn spredning 60_ch000.dat",'rb',) as datFile:
    #print([data.split() for data in datFile])
    X = []
    x0 = []
    for i in datFile:
        #print(i)
        x = int.from_bytes(i[:8],'little')
        X.append(x)
        x0.append(i)
    
    # print(X)
    # print(np.max(X))
    # print(len(X))
    # #print(X)
    # print(int.from_bytes(b'1001000','big'))
