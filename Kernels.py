
# coding: utf-8

# In[1]:

import numpy as np

class Kernel:
    def Eval(x1, x2):
        pass
    def Eval(x):
        pass
    
class LinearKernel(Kernel):
    def Eval(x1, x2):
        return np.dot(x1, x2)
    def Eval(x):
        return np.linalg.norm(x)

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        Kernel.__init__(self)
        self.m_sigma = sigma
    def Eval(x1, x2):
        return np.linalg.norm(np.exp(-self.m_sigma*(x1-x2)))
    def Eval(x):
        return 1.0
    
class IntersectionKernel(Kernel):
    def Eval(x1, x2):
        return np.minimum(x1, x2).sum()
    def Eval(x):
        return np.sum(x)

#incomplete
class Chi2Kernel(Kernel):
    def Eval(x1, x2):
        pass
    def Eval(x):
        return 1.0

#incomplete    
class MultiKernel(Kernel):
    def __init__(self, kernels, featureCounts):
        Kernel.__init__(self)
        self.m_n, self.m_norm, self.m_kernels, self.m_counts = kernels.size(), 1.0/kernels.size(), kernels, featureCounts
    def Eval(x1, x2):
        total, start = 0.0, 0.0
        for i in range(self.m_n):
            c = self.m_counts[i]
            #total += self.m_norm * self.m_kenerls[i].Eval()
        

