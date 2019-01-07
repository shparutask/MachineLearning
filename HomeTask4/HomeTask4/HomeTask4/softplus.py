import numpy as np
import module as mod

class SoftPlus(mod.Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.log(np.ones(input.shape) + np.exp(input))
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.ones(input.shape)/(np.ones(input.shape) + np.exp(-gradOutput))
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"