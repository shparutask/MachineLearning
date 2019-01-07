import numpy as np

def sgd_momentum(x, dx, config, state):
    #"""
    #    This is a very ugly implementation of sgd with momentum 
    #    just to show an example how to store old grad in state.
    #    
    #    config:
    #        - momentum
    #        - learning_rate
    #    state:
    #        - old_grad
    #"""
    
    # x and dx have complex structure, old dx will be stored in a simpler one
    state.setdefault('old_grad', {})
    
    i = 0 
    for cur_layer_x, cur_layer_dx in zip(x,dx): 
        for cur_x, cur_dx in zip(cur_layer_x,cur_layer_dx):
            
            cur_old_grad = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            
            np.add(config['momentum'] * cur_old_grad, config['learning_rate'] * cur_dx, out = cur_old_grad)
            
            cur_x -= cur_old_grad
            i += 1