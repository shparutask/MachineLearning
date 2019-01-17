import numpy as np

import module as mod

class LeakyReLU(mod.Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        # Your code goes here. #
        self.output = np.where(input > 0, input, input * (self.slope - 1) + 1)
        ###############################################

        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. # 
        self.gradInput = np.where(input > 0, gradOutput,  input*(self.slope - 1) + 1)
        ###############################################

        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"