import numpy as np

import module as mod

class Sequential(mod.Module):
         #This class implements a container, which processes `input` data sequentially. 
         
         #`input` is processed by each module (layer) in self.modules consecutively.
         #The resulting array is called `output`. 
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        #Adds a module to the container.
     
        self.modules.append(module)

    def updateOutput(self, input):
        #Basic workflow of FORWARD PASS:
        
        #    y_0    = module[0].forward(input)
        #    y_1    = module[1].forward(y_0)
        #    ...
        #    output = module[n-1].forward(y_{n-2})   
                        
        #Just write a little loop.
       
        # Your code goes here. #
        self.output = self.modules[0].forward(input)

        for i in range(1, len(self.modules)):
            self.output = self.modules[i].forward(self.output)
        ###############################################

        return self.output
 
    def backward(self, input, gradOutput):
        #Workflow of BACKWARD PASS:
        #    
        #    g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
        #    g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
        #    ...
        #    g_1 = module[1].backward(y_0, g_2)   
        #    gradInput = module[0].backward(input, g_1)   
             
             
        #!!!
                
        #To ech module you need to provide the input, module saw while forward pass, 
        #it is used while computing gradients. 
        #Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass) 
        #and NOT `input` to this Sequential module. 
        
        #!!!
        
        # Your code goes here. #
        n = len(self.modules)

        self.gradInput = self.modules[n - 1].backward(self.modules[n - 2].output, gradOutput)

        for i in range(n - 2, 0, -1):
            self.gradInput = self.modules[i].backward(self.modules[i - 1].output, self.gradInput)

        self.gradInput = self.modules[0].backward(input, self.gradInput)        
        ###############################################

        return self.gradInput
      
    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        #Should gather all parameters in a list.
        
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
       #Should gather all gradients w.r.t parameters in a list.
        
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)
    
    def train(self):
        #Propagates training parameter through all modules
        
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        #Propagates training parameter through all modules
        
        self.training = False
        for module in self.modules:
            module.evaluate()