# 3. Import libraries and modules

# next: plot training, validation accuracies
# then select data to start transfer learning process

# recognizes all digits, except hard time with 7 unless horizontal line through it
# next step: try limited transfer learning? 

import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D,Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# needed to avoid error: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

imgloaded=load_img('img/seven.png', target_size=(28,28))  # this is a PIL image
imgloaded=imgloaded.convert('L') #makes it greyscale
x = img_to_array(imgloaded)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape(1,28,28,1)  # this is a Numpy array with shape (1, 3, 150, 150)
x /= 255
x = 1-x   # it was inverted
 
# 4. Load pre-shuffled MNIST data into train and test sets
# not clear how to specify train vs. test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
# 5. Preprocess input data
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

img_cols=28
img_rows=28


# dimension order is different between tf and th, and this fixes it for the Convolution command
if K.image_data_format=='channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1,img_cols,img_rows)
    X_test = X_test.reshape(X_test.shape[0], 1,img_cols,img_rows)
    input_shape = (1,img_cols,img_rows)
else:
    X_train = X_train.reshape(X_train.shape[0],img_cols,img_rows,1)
    X_test = X_test.reshape(X_test.shape[0],img_cols,img_rows,1)
    input_shape = (img_cols,img_rows,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 
# 7. Define model architecture
model = Sequential()

# seems like specifying directly number of filters as first argument
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# checkpoint
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

# 9. Fit model on training data
if True:
	model = load_model('my_model.h5')
else:
	history=model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, validation_data = (X_test, Y_test), verbose=1)
	#model.save('my_model.h5')
	model.save('my_model_v2.h5')  # do not overwrite previous trained model

# 10. Evaluate model on test data (how different from validation in epochs?)
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
result=model.predict_classes(x)
print(result)
#plt.imshow(X_test[0,:,:,0])
result=model.predict_proba(x)
print(result)


# for plotting training progress:
# fit model
# history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=50, verbose=0)
# # evaluate the model
# _, train_acc = model.evaluate(trainX, trainy, verbose=0)
# _, test_acc = model.evaluate(testX, testy, verbose=0)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# # learning curves of model accuracy
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()


