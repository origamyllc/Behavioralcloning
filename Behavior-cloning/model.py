import csv
import cv2 
import numpy as np 
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Lambda
lines = [] 
with open('../behavior-cloning/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
images = []
measurements = []
for line in lines:
  source_path = line[0]
  if (source_path != 'center'):
    filename = source_path.split('/')[-1]
    current_path = '../behavior-cloning/IMG/'+filename
    image = cv2.imread(current_path)
    #crop the image
    cropped_image = image[50:140,:,:]
    #scale to nvidia size
    scaled_image = cv2.resize(cropped_image,(200, 66), interpolation = cv2.INTER_AREA)
    #convert to YUV color space (as nVidia paper suggests)
    img = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2YUV)
    images.append(img)
    measurement = float(line[3])
    measurements.append(measurement) 
   
X = np.array(images)
y = np.array(measurements)

from keras.models import Sequential 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 64
num_classes = 10
epochs = 2

#test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#get the input shape
input_shape = X_train.shape[1:]
print('input shape',input_shape)

model = Sequential()

#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))

model.add(Conv2D(16, (3, 3), padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))

model.save('convolutional.h5')
