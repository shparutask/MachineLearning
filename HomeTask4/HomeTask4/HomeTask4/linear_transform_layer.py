import numpy as np
import module as mod


#input:  batch_size x n_feats1
#output: batch_size x n_feats2

class Linear(mod.Module):
    #A module which applies a linear transformation 
    #A common name is fully-connected layer, InnerProductLayer in caffe. 
    
    #The module should work with 2D input of shape (n_samples, n_feature).
    
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        self.output = np.add(np.dot(self.W, input).reshape(self.b.shape), self.b)
        self.output = self.output.reshape(self.output.shape[0], 1)
        return self.output

    def updateGradInput(self, input, gradOutput):       
        self.gradInput = np.dot(gradOutput, self.gradW)
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradW = input.reshape(input.shape[0], -1).dot(gradOutput).T
        self.gradb = np.sum(gradOutput, axis=0)
        return self.gradW, self.gradb
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q