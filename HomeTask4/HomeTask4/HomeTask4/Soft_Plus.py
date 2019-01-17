import numpy as np

import module as mod

class SoftPlus(mod.Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        # Your code goes here. #
        self.output = np.multiply(input, np.log(np.ones(input.shape) + np.exp(input)))
        ###############################################
        
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. #
        self.gradInput = np.multiply(gradOutput, np.ones(input.shape)/(np.ones(input.shape) + np.exp(-gradOutput)))
        ###############################################
        
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"