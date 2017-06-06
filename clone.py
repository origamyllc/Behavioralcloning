import csv
import cv2 
import numpy as np 

lines = [] 
with open('../behavior-cloning-data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
images = []
measurements = []
for line in lines:
  source_path = line[0]
  if (source_path != 'center'):
    filename = source_path.split('/')[-1]
    current_path = '../behavior-cloning-data/IMG/'+filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement) 

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential 
from keras.layers import Flatten,Dense,Lambda,Input

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5 ,input_shape=(160,320,3)))
#model.add(Lambda(preprocess),output_shape=(160,320,3))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=2)
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
