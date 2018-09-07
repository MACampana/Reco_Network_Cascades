import numpy as np
import pandas as pd
import csv
import keras
#from keras.layers import Dense, Flatten, Conv3D, Dropout, BatchNormalization, Activation, MaxPooling3D
from keras.models import Sequential, load_model
import matplotlib.pylab as plt
#from keras import backend as K

# Load True Values to Predict
Trus = np.load('M:/Test/save_y_all_justE_7kPRED.npy')
loc1 = 'M:/Test/Save_Preds.txt'

batch_size = 1

# Load Saved Model
filepath = 'M:/Test/Save_Model_Best'
model = load_model(filepath)

# Load Input Data to Predict
x_test = np.load('M:/Test/save_x_all_7kPRED.npy')

# Predict
preds = model.predict(x_test, verbose=1, batch_size=batch_size)

# Save Prediction Values
np.savetxt(loc1, preds)

# Open Prediction Values and Format Accordingly
s = open(loc1).read()
s = s.replace('[', '')
s = s.replace(']', '')
f = open(loc1, 'w')
f.write(s)
f.close()

# Read predictions into DataFrame
res = pd.read_csv(loc1, delimiter='\s+', header=None)
#print Trus
#print res
res.columns = ['E']
Epred = res['E'].values
#Zpred = res['Z'].values
#Apred = res['A'].values

# Put True values into List
Etru = []
#Ztru = []
#Atru = []
for i in range(len(Trus)):
    Etru.append(Trus[i][0])
   # Ztru.append(Trus[i][1])
   # Atru.append(Trus[i][2])

'''
# For plotting with cutoff
Etru2 = []
Epred2 =[]
for i in range(len(Etru)):
    if Etru[i] < 55000:
        Etru2.append(Etru[i])
        Epred2.append(Epred[i])
'''
Etru2 = Etru
Epred2 = Epred

# List of True-Predict Differences
difE = []
#difZ = []
#difA = []
for i in range(len(Etru2)):
    difE.append(Etru2[i]-Epred2[i])
    # difZ.append(Ztru[i] - Zpred[i])
    # difA.append(Atru[i] - Apred[i])

# Scatter Plot
plt.scatter(Etru2, Epred2)
plt.xlabel("True Energies (GeV)")
plt.ylabel("Recon Energies (GeV)")
plt.show()

# Histogram of Differences
plt.clf()
plt.hist(difE, bins=50)
plt.xlabel("Difference")
plt.show()

# 2d Histogram
plt.clf()
plt.hist2d(Etru2,Epred2,bins=500)
plt.xlabel("True Energies (GeV)")
plt.ylabel("Recon Energies (GeV)")
plt.xscale('log', basex=10)
plt.yscale('log',basey=10)
plt.show()

# 2d Histogram with More bins
plt.clf()
plt.hist2d(Etru2,Epred2,bins=1000)
plt.xlabel("True Energies (GeV)")
plt.ylabel("Recon Energies (GeV)")
plt.xscale('log', basex=10)
plt.yscale('log',basey=10)
plt.show()
