import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
import os.path

from sklearn.preprocessing import StandardScaler

scaler_save_filename = "myscaler.save"

if os.path.isfile(scaler_save_filename):
    scaler = joblib.load(scaler_save_filename) 
#joblib.dump(scaler1, "myscaler.save") 

needDump = False
if not 'scaler' in locals():
    print("scaler save does not exist")
    scaler = StandardScaler()
    needDump = True


old_data_series = np.linspace(1, 10, 10).reshape(-1, 1)


scaler.fit(old_data_series)

if needDump:
    joblib.dump(scaler, scaler_save_filename)
    
scaled_data_old = scaler.transform(old_data_series)

#print(scaled_data_old)

new_data_series = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2], dtype = "float64").reshape(-1, 1)

#print(new_data_series)


scaled_data_new = scaler.transform(new_data_series)
#print(scaled_data_new)

d = {"old": old_data_series.reshape(-1), "new": scaled_data_old.reshape(-1)}
old_data_dataframe = pd.DataFrame(data = d)

new_data_dataframe = pd.DataFrame(data = {"old": new_data_series.reshape(-1), "new": scaled_data_new.reshape(-1)})

print(old_data_dataframe)
print(new_data_dataframe)

print("get values1: ")

print(new_data_dataframe.iloc[:, 0:2])

print("get values2: ")
print(new_data_dataframe.iloc[:, 0:2].values)

#plt.scatter(x = old_data_series, y = scaled_data_old, color = "red", marker = "x", label = "old")
#plt.scatter(x = new_data_series, y = scaled_data_new * 2, color = "blue", label = "new")
#plt.legend(frameon = False)
#plt.show()
