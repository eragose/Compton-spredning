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