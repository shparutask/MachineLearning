import numpy as np

import module as mod

#input:  batch_size x n_feats
#output: batch_size x n_feats

class SoftMax(mod.Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        
        # Your code goes here. #
        self.output = (np.exp(self.output) / np.sum(np.exp(self.output), axis=1, keepdims=True))
        ###############################################

        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. #
        #dSoftMax = np.diagflat(s) - np.dot(s, s.T)
        dSoftMax = np.matmul(self.output.reshape(input.shape[0], input.shape[1], 1), np.ones((input.shape[0], 1, input.shape[1])))
        dSoftMax = np.matmul(np.eye(input.shape[1]).reshape(input.shape[1], input.shape[1], 1), np.ones((input.shape[1], 1, input.shape[0]))).T - dSoftMax
        dSoftMax = np.multiply(np.matmul(np.ones((input.shape[0], input.shape[1], 1)), self.output.reshape(input.shape[0], 1, input.shape[1])), dSoftMax)
        
        self.gradInput = np.matmul(gradOutput.reshape(input.shape[0], 1, input.shape[1]), dSoftMax).reshape(input.shape)
        ###############################################
        
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"