import numpy as np
import pandas as pd
import glob
import os

folder = 'kalibrering'
seperator = ' '
file_type = '*.txt'



read_files = glob.glob(os.path.join(folder, file_type))

#Creates a list of all file names
np_array_values = []
for files in read_files:
    pdfile = pd.read_csv(files, sep=seperator, skiprows=3)           #Specify seperator
    np_array_values.append(pdfile)

print(np_array_values[0])
