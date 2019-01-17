import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn.metrics import accuracy_score

# Optimizer params
optimizer_config = {'learning_rate' : 1e-1, 'momentum': 0.9}
optimizer_state = {}

def sgd_momentum(x, dx, config, state):
        #This is a very ugly implementation of sgd with momentum 
        #just to show an example how to store old grad in state.
        
        #config:
        #    - momentum
        #    - learning_rate
        #state:
        #    - old_grad
    
    # x and dx have complex structure, old dx will be stored in a simpler one
    state.setdefault('old_grad', {})
    
    i = 0 
    for cur_layer_x, cur_layer_dx in zip(x,dx): 
        for cur_x, cur_dx in zip(cur_layer_x,cur_layer_dx):
            
            cur_old_grad = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            
            np.add(config['momentum'] * cur_old_grad, config['learning_rate'] * cur_dx, out = cur_old_grad)
            
            cur_x -= cur_old_grad
            i += 1

# batch generator
def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start:end]
    
        yield X[batch_idx], Y[batch_idx]

def train_network(X, Y, net, criterion, n_epoch, batch_size):       
    loss_history = []
    for i in range(n_epoch):
        for x_batch, y_batch, in get_batches((X, Y), batch_size):
            
            net.zeroGradParameters()
        
            # Forward
            predictions = net.forward(x_batch)
            loss = criterion.forward(predictions, y_batch)

            # Backward
            dp = criterion.backward(predictions, y_batch)
            net.backward(x_batch, dp)
        
            # Update weights
            sgd_momentum(net.getParameters(), 
                     net.getGradParameters(), 
                     optimizer_config,
                     optimizer_state)      
        
        loss_history.append(loss)

        print('Current loss: ')
        print(loss)
       
    # Visualize
    display.clear_output(wait=True)
    plt.figure(figsize=(8, 6))
        
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.plot(loss_history, 'b')
    plt.show()

    
def test_network(X, Y, net, criterion, n_epoch, batch_size):       
    loss_history = []
    acc_history = []
    for i in range(n_epoch):
        for x_batch, y_batch, in get_batches((X, Y), batch_size):
            
            net.zeroGradParameters()
        
            # Forward
            predictions = net.forward(x_batch)                       
            acc = accuracy_score(np.argmax(y_batch, axis=1), np.argmax(predictions, axis=1))
        
        acc_history.append(acc)

        print('Current accuracy: ')
        print(acc)
       
    # Visualize
    display.clear_output(wait=True)
    plt.figure(figsize=(8, 6))
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.plot(acc_history, 'b')
    plt.show()
