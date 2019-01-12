import linear_transform_layer as lin
import softmax as lsm
import sequence_container as seq
import ReLU as r
import negative_logLikelihood_criterion_numstable as nlls
import dropout as drop
import numpy as np
import optimizer as opt
from IPython import display
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Optimizer params
optimizer_config = {'learning_rate' : 1e-1, 'momentum': 0.9}
optimizer_state = {}

#Basic training loop. Examine it.
loss = 0

# batch generator
def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(batch_size):
        #end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start]
    
        yield X[batch_idx], Y[batch_idx]

def loss_acc_net(network, batch_size, n_epoch, X, Y, criterion):
    losses = ""
    accs = ""
    for i in range(n_epoch):
        loss_history = []
        acc = 0
        j = 0
        for x_batch, y_batch in get_batches((X, Y), batch_size):
            network.zeroGradParameters()
        
            # Forward
            predictions = network.forward(x_batch)
            loss = criterion.forward(predictions, y_batch)
            pred = list(predictions).index(max(predictions))
            
            if(pred == y_batch):
                acc+=1

            # Backward
            dp = criterion.backward(predictions, y_batch)
            pred = network.backward(x_batch, dp)

            par = network.getParameters()
            gradPar = network.getGradParameters()

            # Update weights
            opt.sgd_momentum(par, 
                     gradPar, 
                     optimizer_config,
                     optimizer_state)      
        
            loss_history.append(abs(loss)) 
            j+=1

            display.clear_output(wait=True)
            plt.figure(figsize=(8, 6))
        
            plt.title("Training loss")
            plt.xlabel("#iteration")
            plt.ylabel("loss")
            plt.plot(loss_history, 'b')
            plt.show()

        losses += str(round(np.mean(np.array(loss_history)), 2)) + ";\t"
        accs += str(round(acc/60000, 10)) + ";\t"
    return losses, accs

def get_Networks():    
    criterion = nlls.ClassNLLCriterion()

    net_3layer_dropout = seq.Sequential()
    net_3layer_dropout.add(lsm.SoftMax())
    net_3layer_dropout.add((drop.Dropout()))
    net_3layer_dropout.add(lin.Linear(784, 10))

    net_2layer_dropout = seq.Sequential()
    net_2layer_dropout.add(lsm.SoftMax())
    net_2layer_dropout.add((drop.Dropout()))

    net_2layer = seq.Sequential()
    net_2layer.add(lsm.SoftMax())
    net_2layer.add(lin.Linear(784, 10))
    
    net_3layer = seq.Sequential()
    net_3layer.add(r.ReLU())
    net_3layer.add(lsm.SoftMax())
    net_3layer.add(lin.Linear(784, 10))
    
    return criterion, net_3layer, net_3layer_dropout, net_2layer, net_2layer_dropout