import numpy as np

import module as mod

#input:  batch_size x n_feats
#output: batch_size x n_feats

class BatchNormalization(mod.Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.5):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0 
        self.moving_variance = 0
        
    def updateOutput(self, input):
        # Your code goes here. #
        if(self.training):
            batch_mean = np.mean(input, axis=0)
            batch_variance = np.var(input, axis=0)   
                    
            self.output = np.subtract(input, batch_mean)/np.sqrt(batch_variance + self.EPS)
            self.moving_mean = self.moving_mean * self.alpha + batch_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + batch_variance * (1 - self.alpha)
        else:
            self.output = np.subtract(input, self.moving_mean)/self.moving_variance
        ###############################################
        # use self.EPS please
        
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. #        
        batch_mean = np.mean(input, axis=0)
        batch_variance = np.var(input, axis=0)
        
        inputGrad = gradOutput / np.sqrt(batch_variance + self.EPS)
        meanGrad = -np.sum(inputGrad, axis = 0) / input.shape[0]
        varGgrad = -np.subtract(input, batch_mean)/ input.shape[0] * np.sum(gradOutput * np.divide(input - batch_mean,(batch_variance + self.EPS)**1.5), axis=0)  
        
        self.gradInput = inputGrad + meanGrad + varGgrad  
        ###############################################

        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"

class ChannelwiseScaling(mod.Module):
       #Implements linear transform of input y = \gamma * x + \beta
       #where \gamma, \beta - learnable vectors of length x.shape[-1]
    
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"