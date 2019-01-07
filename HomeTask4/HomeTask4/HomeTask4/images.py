import numpy as np
import gzip
import matplotlib.pyplot as plt
from IPython import display

import linear_transform_layer as lin
import logSoftMax as lsm
import sequence_container as seq
import ReLU as r
import negative_logLikelihood_criterion_numstable as nlls
import optimizer as opt
import dropout as drop

FILE_TRAIN_LABELS_PATH = "../Dataset/train-labels-idx1-ubyte.gz"
FILE_TRAIN_IMAGES_PATH = "../Dataset/train-images-idx3-ubyte.gz"

FILE_TEST_LABELS_PATH = "../Dataset/t10k-labels-idx1-ubyte.gz"
FILE_TEST_IMAGES_PATH = "../Dataset/t10k-images-idx3-ubyte.gz"

f = gzip.open(FILE_TRAIN_IMAGES_PATH,'r')
f.read(16)
num_train_images = 60000
count_train_image_rows = 28
count_train_image_cols = 28
buf = f.read(count_train_image_rows * count_train_image_cols * num_train_images)
train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
train_data = train_data.reshape(num_train_images, count_train_image_rows * count_train_image_cols, 1)

train_labels = []

f = gzip.open(FILE_TRAIN_LABELS_PATH,'r')
f.read(8)
num_train_labels = 60000
for i in range(0, num_train_labels):
    buf = f.read(1)
    train_labels.append((np.frombuffer(buf, dtype=np.uint8).astype(np.int64)))

f = gzip.open(FILE_TEST_IMAGES_PATH,'r')
f.read(16)
num_test_images = 10000
count_test_image_rows = 28
count_test_image_cols = 28
buf = f.read(count_test_image_rows * count_test_image_cols * num_test_images)
test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test_data = test_data.reshape(num_test_images, count_test_image_rows * count_test_image_cols, 1)

test_labels = []

f = gzip.open(FILE_TEST_LABELS_PATH,'r')
f.read(8)
num_test_labels = 10000
for i in range(0, num_test_labels):
    buf = f.read(1)
    test_labels.append((np.frombuffer(buf, dtype=np.uint8).astype(np.int64)))

# Generate some data
N = num_train_images

Y = train_labels
X = train_data

#Define a logistic regression for debugging.

net = seq.Sequential()
net.add(lin.Linear(784, 2))
net.add(lsm.LogSoftMax())

criterion = nlls.ClassNLLCriterion()

print(net)

# Test something like that then 

net.add(lin.Linear(2, 1)) 
net.add((drop.Dropout()))

#Start with batch_size = 1000 to make sure every step lowers the loss, then try stochastic version.

# Optimizer params
optimizer_config = {'learning_rate' : 1e-1, 'momentum': 0.9}
optimizer_state = {}

# Looping params
n_epoch = 20
batch_size = 128

#Basic training loop. Examine it.

loss_history = []
loss = 0

def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):        
        batch_idx = indices[start]
    
        yield X[batch_idx], Y[batch_idx]

for i in range(n_epoch):
    for x_batch, y_batch in get_batches((X, Y), batch_size):        
        net.zeroGradParameters()
        
        # Forward
        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)
    
        # Backward
        dp = criterion.backward(predictions, y_batch)
        net.backward(x_batch, dp)

        # Update weights
        opt.sgd_momentum(net.getParameters(), 
                     net.getGradParameters(), 
                     optimizer_config,
                     optimizer_state)      
        
        loss_history.append(loss)

    # Visualize
    display.clear_output(wait=True)
    plt.figure(figsize=(8, 6))
        
    plt.title("Training loss")
    plt.xlabel("#iteration")
    plt.ylabel("loss")
    plt.plot(loss_history, 'b')
    plt.show()
    
    print('Current loss: ')
    print(loss)


