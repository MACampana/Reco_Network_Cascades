import numpy as np
import pandas as pd
import csv
import keras
from keras.layers import Dense, Flatten, Conv3D, Dropout, BatchNormalization, Activation, MaxPooling3D
from keras.models import Sequential, load_model
import matplotlib.pylab as plt
from keras import backend as K

# Load Data Files (x = train, y = validate)
x = np.load('M:/Test/save_x_all.npy')
y = np.load('M:/Test/save_y_all_justE.npy')


# =====================================================================================================================
# For Reproducibility
seed = 8
np.random.seed(seed)

# Set Constant Network Properties
num_classes = 1 #1 for energy, 3 for energy/zenith/azimuth
input_shape = (10, 10, 60, 7)
epochs = 80
batch_size = 5

# Build Model
K.set_image_data_format('channels_last')

model = Sequential()

model.add(Conv3D(filters=50, kernel_size=2, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

model.add(Conv3D(filters=50, kernel_size=2))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss=keras.losses.mean_squared_logarithmic_error,
              optimizer=keras.optimizers.SGD(lr=0.0001,nesterov=True))

# keras.optimizers.SGD(lr=0.01,momentum=0.8,decay=0.1,nesterov=False)
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Save Model if better loss than previous Epoch
filepath = 'M:/Test/Save_Model_Best'
#model = load_model(filepath)
chkpt = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                        save_weights_only=False, mode='auto', period=1)

# Fit Model and validate
history = model.fit(x, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    verbose=1,
                    validation_split=.10,
                    callbacks=[chkpt])

# Save final model version
model.save('M:/Test/Saved_Model_End')

# Plot Losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
