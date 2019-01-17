import numpy as np

import module as mod

#input:  batch_size x n_feats
#output: batch_size x n_feats

class Dropout(mod.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        # Your code goes here. #
        if(self.training):
            self.mask = (np.random.rand(*input.shape) < self.p) / self.p
            self.output = np.multiply(input, self.mask)
        else:
            self.output = input
        ###############################################

        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. #
        if(self.training):
            self.gradInput = input * self.mask
        else:
            self.gradInput = input
        ###############################################

        return self.gradInput
        
    def __repr__(self):
        return "Dropout"