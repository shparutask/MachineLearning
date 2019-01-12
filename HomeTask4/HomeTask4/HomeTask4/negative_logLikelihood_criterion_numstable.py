import numpy as np
import criterion as crit

class ClassNLLCriterion(crit.Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target): 
        self.output = -np.sum(np.dot(input, target))/target.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = np.sum(np.dot(np.ones(input.shape), target))/target.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"