import numpy as np
import pandas as pd
import csv

# Load hit DOM data into DataFrame without True Values
loc = 'M:/Test/Data_ME_2013_10to500_withTest_0001_CUT.txt'
frame1 = pd.read_csv(loc, sep='\s+', comment='E', header=None) #dom hit data
frame1.columns = ['string','z','qSum','#pulses','pulse1time','pulseLtime','pulseAVGtime','pulsetimeSTDDEV',
                 'pulseMAXq']

# Load conversion for DOM and string to grid coordinate
# Also, reorder columns and set string number as index
loc2 = 'M:/Test/Grid_Transformation.txt'
frame2 = pd.read_csv(loc2, sep='\s+', header=None) #Grid Data
frame2.columns = ['string','y','x']
frame2 = frame2[['x','y','string']]
frame2.set_index(['string'], inplace=True)

# Combine Hit DOM data with grid coordinates 
# Coordinates without DOM data will be zeros 
strings = frame1['string'].values
blank = []
zero = np.zeros([10,10,60,7])
#E = 0

for i in range(len(frame1.index)):
    if i == 0: # For First DOM data in file
        x = frame2.loc[strings[i]]['x']             # get x coordinate from string number
        y = frame2.loc[strings[i]]['y']             # get y coordinate from string number
        z = frame1.iloc[i, 1]                       # get z coordinate from DOM number
        zero[x][y][z-1] = frame1.iloc[i, 2:].values # add input data in corresponding coordinates of zero array
    elif strings[i] >= strings[i - 1]:              # Data sorted by string number, if FALSE means next event starts
        x = frame2.loc[strings[i]]['x']
        y = frame2.loc[strings[i]]['y']
        z = frame1.iloc[i, 1]
        zero[x][y][z-1] = frame1.iloc[i, 2:].values
    else:                                           # MISTAKE NOTICED: Skips first hit DOM in each event after first
        blank.append(zero) #[2:8,2:8,0:30]          # Add data for this event to blank array
        zero = np.zeros([10,10,60,7])               # FIX: Add in commented lines here as follows...
        #x = frame2.loc[strings[i]]['x']
        #y = frame2.loc[strings[i]]['y']
        #z = frame1.iloc[i, 1]
        #zero[x][y][z-1] = frame1.iloc[i, 2:].values
    print i,"/",len(frame1.index)-1                 # For viewing progress
blank.append(zero) #[2:8,2:8,0:30]                  # Add last event data to combined array

frame2 = frame2.iloc[0:0]                           # Clear data no longer needed to conserve MEMORY
frame1 = frame1.iloc[0:0]
zero = np.delete(zero, np.s_[::])

#blank = np.array(blank, copy=False)
#print blank.shape

np.save('M:/Test/save_x_all_7kPRED.npy', blank)     # Save input data array

#s2 = len(blank) / 10

#x_test = blank[:s2]
#x_train = blank[s2:]

#np.save('M:/Test/save_x_train.npy', x_train)
#np.save('M:/Test/save_x_test.npy', x_test)

#blank = np.delete(blank, np.s_[::])
#x_test = np.delete(x_test, np.s_[::])
#x_train = np.delete(x_train, np.s_[::])
                                                    # Delete to conserve memory
del blank[:]
#del x_test[:]
#del x_train[:]

#--------TRUE---------
# Load data and select only True value lines
rows = csv.reader(open(loc, 'rb'), delimiter='\t')
arows = [row for row in rows for element in row if 'E' in element]

# Create DataFrame of True Values
row_frame = pd.DataFrame(arows)
row_frame.columns = ['EventNum','Energy','Zenith','Azimuth','X','Y','Z']
row_frame.drop(labels=['EventNum','X','Y','Z'], axis=1, inplace=True)     # Drop Columns that will not be used (For Now)
row_frame = row_frame.astype(np.float32, copy=False)
row_frame_np = row_frame.values
print row_frame_np.shape

# Save True Values array (Energy, Zenith, Azimuth)
np.save('M:/Test/save_y_all_7kPRED.npy', row_frame_np)

row_frame.drop(labels=['Zenith','Azimuth'], axis=1, inplace=True)         # Drop columns that will not be used (For Now)
row_frame_np2 = row_frame.values
print row_frame_np2.shape

# Save True Values (Energy only)
np.save('M:/Test/save_y_all_justE_7kPRED.npy', row_frame_np2)

#s1 = len(row_frame) / 10

#y_test = row_frame[:s1]
#y_train = row_frame[s1:]

#np.save('M:/Test/save_y_train.npy', y_train)
#np.save('M:/Test/save_y_test.npy', y_test)

