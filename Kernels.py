
# coding: utf-8

# In[1]:

import numpy as np

class Kernel:
    def __init__(self):
        pass
    def Eval(self, x1, x2=None):
        pass
    
class LinearKernel(Kernel):
    def __init__(self):
        pass
    def Eval(self, x1, x2=None):
        if x2 != None:
            return np.dot(x1, x2)
        else:
            return np.linalg.norm(x1)

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        Kernel.__init__(self)
        self.m_sigma = sigma
    def Eval(self, x1, x2=None):
        if x2 is not None:
            norm_val = (np.linalg.norm(x1-x2)) ** 2
            # print(-1 * float(self.m_sigma))
            # print(norm_val)
            # print(-1 * float(self.m_sigma) * norm_val)
            # print(np.exp(-1 * float(self.m_sigma) * norm_val))
            return np.exp(-1 * float(self.m_sigma) * norm_val)
            # inner = map(lambda d: -1*float(self.m_sigma)*d, (x1-x2))
            # return np.exp(np.linalg.norm(inner))
        else:
            return 1.0
    
class IntersectionKernel(Kernel):
    def __init__(self):
        pass
    def Eval(self, x1, x2=None):
        if x2 is not None:
            return np.minimum(x1, x2).sum()
        else:
            return np.sum(x1)

#incomplete
class Chi2Kernel(Kernel):
    def __init__(self):
        pass
    def Eval(self, x1, x2=None):
        pass

#incomplete    
class MultiKernel(Kernel):
    def __init__(self, kernels, featureCounts):
        Kernel.__init__(self)
        self.m_n, self.m_norm, self.m_kernels, self.m_counts = kernels.size(), 1.0/kernels.size(), kernels, featureCounts
    def Eval(self, x1, x2):
        total, start = 0.0, 0.0
        for i in range(self.m_n):
            c = self.m_counts[i]
            #total += self.m_norm * self.m_kenerls[i].Eval()
        

