import numpy as np
import module as mod

class ELU(mod.Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        
    def updateOutput(self, input):
        self.output = np.where(input > 0, input, alpha * (np.exp(input) - 1))
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(gradOutput > 0, gradOutput, alpha * (np.exp(gradOutput) - 1))
        return self.gradInput
    
    def __repr__(self):
        return "ELU"