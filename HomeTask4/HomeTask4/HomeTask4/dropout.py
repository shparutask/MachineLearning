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
        if(self.training):
            self.mask = 1 / (1 - self.p)
            self.output = np.where(input > 0, input/(1 - self.p), 0)
        else:
            self.output = input
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, self.mask)
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"