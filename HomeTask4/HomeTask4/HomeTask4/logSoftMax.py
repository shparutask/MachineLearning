import numpy as np
import module as mod
    
#input:  batch_size x n_feats
#output: batch_size x n_feats

class LogSoftMax(mod.Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=0, keepdims=True))

        self.output = np.subtract(input, np.log(np.sum(np.exp(input))))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.subtract(np.ones(input.shape[0]), 1/np.sum(gradOutput))
        self.gradInput = self.gradInput.reshape(1, self.gradInput.shape[0])
        return self.gradInput
    
    def __repr__(self):
        return "LogSoftMax"