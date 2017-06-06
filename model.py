import csv
import cv2 
import numpy as np
import keras
from sklearn.model_selection import train_test_split

lines = []
with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
images = []
measurements = []
for line in lines:
  source_path = line[0]
  if (source_path != 'center'):
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/'+filename
    image = cv2.imread(current_path)
    #crop the image
    cropped_image = image[50:140,:,:]
    #scale to nvidia size
    scaled_image = cv2.resize(cropped_image,(200, 66),  interpolation = cv2.INTER_AREA)
    #convert to YUV color space (as nVidia paper suggests)
    img = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2YUV)
    images.append(img)
    measurement = float(line[3])
    measurements.append(measurement)
    image_flipped = np.fliplr(img)
    images.append(image_flipped)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)
X = X_train
y = y_train
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Lambda,Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
#get the input shape
input_shape = X_train.shape[1:]
print('input shape',input_shape)


batch_size = 128
num_classes = 45
epochs = 4

#test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train -= 0.5
X_test -= 0.5
 

model = Sequential()
model.add(Conv2D(24,5, 5,input_shape=input_shape,subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Conv2D(36,5,5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(Dropout(0.25))
model.add(Conv2D(48,5,5,subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(Dropout(0.25))
model.add(Conv2D(64, 3, 3,border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Conv2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))
model.save('init.h5')
