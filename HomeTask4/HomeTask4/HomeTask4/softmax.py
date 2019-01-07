import numpy as np
import module  as mod

#input:  batch_size x n_feats
#output: batch_size x n_feats

class SoftMax(mod.Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        self.output = np.exp(self.output) / np.sum(np.exp(self.output))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        sum_exp = np.sum(np.exp(input))
        self.gradInput = (sum_exp * input - np.square(input)) / sum_exp ** 2
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"