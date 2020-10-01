from scipy import io as spio
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage

#emnist = spio.loadmat("/Users/alexej/Documents/Data/neural_networks_data/emnist_matlab/emnist-digits.mat")
emnist = spio.loadmat("/Users/alexej/Documents/Data/neural_networks_data/emnist_matlab/emnist-letters.mat")
# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1]
# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]

# store labels for visualization
train_labels = y_train
test_labels = y_test


# next should specify random selection
# select data set
neach=10;
letter_range=10;

# some problem here
selidx=np.zeros(neach*letter_range,dtype=np.int32)

for ii in np.arange(letter_range):
	idx=np.where(y_test==ii+1)[0]   # note that letters start with index 1
	np.random.shuffle(idx)   # does it in place
	np.put(selidx,np.arange(neach)+ii*neach,idx)

#train_mask = np.isin(y_train, [1, 4])
#test_mask = np.isin(y_test, [1, 4])
#arr = np.arange(10)
#

#plt.imshow(np.reshape(x_test[127,:],[28,28]))
#plt.imshow(np.transpose(np.reshape(x_test[idx[0]],[28,28])))
#plt.show()

x_test2=np.transpose(np.reshape(x_test,(np.shape(x_test)[0],28,28)),[0,2,1])
n_images=np.shape(selidx)[0]
fullimg=montage(x_test2[selidx,:,:],grid_shape=(10,10))  # ,,
plt.imshow(fullimg)
plt.show()