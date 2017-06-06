import csv
import cv2 
import numpy as np 
from sklearn.model_selection import train_test_split
import keras

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

X = np.array(images)
y = np.array(measurements)

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop

batch_size = 32
num_classes = 1
nb_epoch = 2

#test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# This is the shape of the image
input_shape = X_train.shape[1:]
print(input_shape, 'input shape')


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# number of convolutional filters to use
nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2

# size of pooling area for max pooling
pool_size = (2, 2)

# convolution kernel size
kernel_size = (3, 3)

# Initiating the model
model = Sequential()

# Starting with the convolutional layer
# The first layer will turn 1 channel into 16 channels
model.add(Conv2D(nb_filters1, kernel_size[0], kernel_size[1],input_shape=input_shape))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 16 channels into 8 channels
model.add(Conv2D(nb_filters2, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 8 channels into 4 channels
model.add(Conv2D(nb_filters3, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 4 channels into 2 channels
model.add(Conv2D(nb_filters4, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=pool_size))
# Apply dropout of 25%
model.add(Dropout(0.25))

# Flatten the matrix. The input has size of 360
model.add(Flatten())
# Input 360 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Apply dropout of 50%
model.add(Dropout(0.5))
# Input 16 Output 1
model.add(Dense(num_classes))

# Compile model using Adam optimizer 
# and loss computed by mean squared error
model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

### Model training
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('convolutional.h5')
