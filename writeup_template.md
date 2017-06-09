# Files Submitted 
#### 1. includes the following files
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* init.h5 containing a trained convolution neural network 
* run1.mp4 containing a video of the output
* video.py containing a script to create a video
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Running the code
The car can be driven autonomously around the track by executing 
```sh
python drive.py init.h5
```
# Model Architecture and Training Strategy
## 1. Data Preprocessing 

## 2. Model Architecture

In order to build the CNN required for the self driving vehcle I ended up using a variation of the architecture as explained in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ 


#### The CNN has the following architecture 

| Layers        | Layer Type | Filter size   | deapth        | output shape |
| ------------- |:-------------:|:-------------:| -------------:| -------------:|
| **Layer1**    | input          |None          | 3             | 66 X 200 | 
| **Layer2**    | convolutional  |5 X 5         | 24            | 31 X 98 |
| **Layer3**    | convolutional  |5 X 5         | 36            | 14 X 47 |
| **Layer4**    | convolutional  |3 X 3         | 48            | 5 X 22 |
| **Layer5**    | convolutional  |3 X 3         | 64            | 3 X 20 |
| **Layer6**    | convolutional  |3 X 3         | 64            | 1 X 18 |
| **Layer7**    | fully connected|None          | 100           | ( 1 x 18 )  X 100 |
| **Layer9**    | fully connected|None          | 50            | ( 1 x 18 X 100 )  X 50 |
| **Layer10**   | fully connected|None          | 10            |  ( 1 x 18 X 100 X 50 )  X 10|
| **Layer11**   | Flatten        |None          | 10            | |
| **Layer12**   | fully connected|None          | 1             | (1 x 18 X 100 X 50 X 10 ) X 1 |

after each layer I used an ELU(relu) activation function for further information on ELU activation refer 
https://keras.io/layers/advanced-activations/#elu

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

## 3. Changing drive.py to work with new image size

## 4. Creating the output video
