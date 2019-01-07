import numpy as np

class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
            #Given an input and a target, compute the loss function 
            #associated to the criterion and return the result.
            
            #For consistency this function should not be overrided,
            #all the code goes in `updateOutput`.
       
        return self.updateOutput(input, target)

    def backward(self, input, target):
            #Given an input and a target, compute the gradients of the loss function
            #associated to the criterion and return the result. 

            #For consistency this function should not be overrided,
            #all the code goes in `updateGradInput`.
        
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        #Function to override.
        
        return self.output

    def updateGradInput(self, input, target):
        #Function to override.
        
        return self.gradInput   

    def __repr__(self):
        #Pretty printing. Should be overrided in every module if you want 
        #to have readable description. 
        
        return "Criterion"