import numpy as np
import module  as mod

#input:  batch_size x n_feats
#output: batch_size x n_feats

class SoftMax(mod.Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        z = np.subtract(input, input.max(axis=1, keepdims=True))
        z -= np.max(z)
        self.output = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        s = input.reshape(-1,1)
        self.gradInput = np.diagflat(s) - np.dot(s, s.T)  
        
        #self.gradInput = np.diag(input.reshape(input.shape[0]))
          #for i in range(len(self.gradInput)):
          #    for j in range(len(self.gradInput)):
          #        if i == j:
          #            self.gradInput[i][j] = input[i] * (1 - input[i])
          #        else:
          #            self.gradInput[i][j] = -input[i]*input[j]
          #
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"