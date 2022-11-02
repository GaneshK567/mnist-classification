# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
The MNIST dataset is a collection of handwritten digits. The task is to classify a given image
of a handwritten digit into one of 10 classes representing integer values from 0 to 9,
inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here
we build a convolutional neural network model that is able to classify to it’s appropriate
numerical value.

![dataset](https://user-images.githubusercontent.com/64765451/199420039-646093a0-2047-4014-92e6-b122480bc9a8.png)

## Neural Network Model

![nnmodel](https://user-images.githubusercontent.com/64765451/199419996-415bde34-eedd-494d-96bb-f69b8d5a085f.jpg)

## DESIGN STEPS

### STEP 1:
Download and load the dataset
### STEP 2:
Scale the dataset between it’s min and max values
### STEP 3:
Using one hot encode, encode the categorical values
### STEP 4:
Split the data into train and test
### STEP 5:
Build the convolutional neural network model
### STEP 6:
Train the model with the training data
### STEP 7:
Plot the performance plot
### STEP 8:
Evaluate the model with the testing data


## PROGRAM

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets
import mnist import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd from sklearn.metrics
import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
y_train_onehot.shape single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
# Write your code here
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10,activation="softmax"))
model.summary()
# Choose the appropriate parameters
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5, batch_size=64,
validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)4
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img(number2.jpg')
type(img)
img = image.load_img(number2.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax( model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)), axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![plot](https://user-images.githubusercontent.com/64765451/199420456-51f0afb0-bc59-4d10-856c-f53db5a216a0.png)

### Classification Report

![classification report](https://user-images.githubusercontent.com/64765451/199420536-394bc295-1acd-4dc4-bf6c-1b5ebc720dbd.png)

### Confusion Matrix

![confusion matrix](https://user-images.githubusercontent.com/64765451/199420522-edcee17c-2299-4c47-9a47-2a040bd72c8c.png)

### New Sample Data Prediction

![output](https://user-images.githubusercontent.com/64765451/199420500-741c2824-265f-42b7-a4c5-66e3e39ee3e1.png)

## RESULT
Successfully developed a convolutional deep neural network for digit classification and
verified the response for scanned handwritten images
