import numpy as np
import pandas as pd
import csv

loc = 'M:/Test/Data_ME_2013_10to500_withTest_0001_CUT.txt'
frame1 = pd.read_csv(loc, sep='\s+', comment='E', header=None) #dom hit data
frame1.columns = ['string','z','qSum','#pulses','pulse1time','pulseLtime','pulseAVGtime','pulsetimeSTDDEV',
                 'pulseMAXq']

loc2 = 'M:/Test/Grid_Transformation.txt'
frame2 = pd.read_csv(loc2, sep='\s+', header=None) #Grid Data
frame2.columns = ['string','y','x']
frame2 = frame2[['x','y','string']]
frame2.set_index(['string'], inplace=True)

strings = frame1['string'].values
blank = []
zero = np.zeros([10,10,60,7])
#E = 0
#j=1
for i in range(len(frame1.index)):
    if i == 0:
        x = frame2.loc[strings[i]]['x']
        y = frame2.loc[strings[i]]['y']
        z = frame1.iloc[i, 1]
        zero[x][y][z-1] = frame1.iloc[i, 2:].values
    elif strings[i] >= strings[i - 1]:
        x = frame2.loc[strings[i]]['x']
        y = frame2.loc[strings[i]]['y']
        z = frame1.iloc[i, 1]
        zero[x][y][z-1] = frame1.iloc[i, 2:].values
    else:
        blank.append(zero) #[2:8,2:8,0:30]
        #j += 1
        zero = np.zeros([10,10,60,7])

    print i,"/",len(frame1.index)-1
blank.append(zero) #[2:8,2:8,0:30]

frame2 = frame2.iloc[0:0] # Clear data no longer needed
frame1 = frame1.iloc[0:0]
zero = np.delete(zero, np.s_[::])

#blank = np.array(blank, copy=False)
#print blank.shape

np.save('M:/Test/save_x_all_7kPRED.npy', blank)

#s2 = len(blank) / 10

#x_test = blank[:s2]
#x_train = blank[s2:]

#np.save('M:/Test/save_x_train.npy', x_train)
#np.save('M:/Test/save_x_test.npy', x_test)

#blank = np.delete(blank, np.s_[::])
#x_test = np.delete(x_test, np.s_[::])
#x_train = np.delete(x_train, np.s_[::])

del blank[:]
#del x_test[:]
#del x_train[:]

#--------TRUE---------
rows = csv.reader(open(loc, 'rb'), delimiter='\t')
arows = [row for row in rows for element in row if 'E' in element]

row_frame = pd.DataFrame(arows)
row_frame.columns = ['EventNum','Energy','Zenith','Azimuth','X','Y','Z']
row_frame.drop(labels=['EventNum','X','Y','Z'], axis=1, inplace=True)
row_frame = row_frame.astype(np.float32, copy=False)
row_frame_np = row_frame.values
print row_frame_np.shape

np.save('M:/Test/save_y_all_7kPRED.npy', row_frame_np)

row_frame.drop(labels=['Zenith','Azimuth'], axis=1, inplace=True)
row_frame_np2 = row_frame.values
print row_frame_np2.shape

np.save('M:/Test/save_y_all_justE_7kPRED.npy', row_frame_np2)

#s1 = len(row_frame) / 10

#y_test = row_frame[:s1]
#y_train = row_frame[s1:]

#np.save('M:/Test/save_y_train.npy', y_train)
#np.save('M:/Test/save_y_test.npy', y_test)

