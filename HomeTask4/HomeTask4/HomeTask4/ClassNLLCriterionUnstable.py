import numpy as np
from math import log

import Criterion as crit

#input:  batch_size x n_feats - probabilities
#target: batch_size x n_feats - one-hot representation of ground truth
#output: scalar

class ClassNLLCriterionUnstable(crit.Criterion):
    EPS = 1e-15
    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()
        
    def updateOutput(self, input, target): 
        
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Your code goes here. #
        self.output = -np.mean(np.log(input_clamp) * target)
        ###############################################
        
        return self.output

    def updateGradInput(self, input, target):
        
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Your code goes here. #
        self.gradInput = - (target / input_clamp) / input.shape[0]
        ###############################################
        
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterionUnstable"