import module as mod
import numpy as np

class LeakyReLU(mod.Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        self.output = np.where(input > 0, input, slope * input)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, 1, slope)
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"