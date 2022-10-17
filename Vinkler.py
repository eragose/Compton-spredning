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

with open("Comptorn spredning 40_ch000.dat",'rb',) as datFile:
    #print([data.split() for data in datFile])
    for i in datFile:
        #print(int.from_bytes(i[:4],'little'))
        print(len(i.split()))
    X = [data.split('/') for data in datFile]
    #print(X)
