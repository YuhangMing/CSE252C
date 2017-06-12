
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
        Kernel.__init__(self)

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
            # print("x1 - x2")
            # print(x1 - x2)
            # tmp = np.array(x1-x2, dtype = np.float32)
            # norm_val = (np.linalg.norm(x1 - x2, ord=2))
            norm_val = (np.linalg.norm(x1 - x2)) ** 2
            # print(-1 * float(self.m_sigma))
            # print(norm_val)
            return np.exp(-1 * float(self.m_sigma) * norm_val)
            # inner = map(lambda d: -1*float(self.m_sigma)*d, (x1-x2))
            # return np.exp(np.linalg.norm(inner))
        else:
            return 1.0


class IntersectionKernel(Kernel):
    def __init__(self):
        Kernel.__init__(self)

    def Eval(self, x1, x2=None):
        if x2 is not None:
            return np.minimum(x1, x2).sum()
        else:
            return np.sum(x1)


#complete
class Chi2Kernel(Kernel):
    def __init__(self):
        Kernel.__init__(self)

    def Eval(self, x1, x2=None):
        if x2 is not None:
            result = 0.0
            for i in range(len(x1)):
                a, b = x1[i], x2[i]
                result += (a-b)*(a-b)/(0.5*(a+b)+1e-8)
            return 1.0 - result
        else:
            return 1.0



#complete    
class MultiKernel(Kernel):
    def __init__(self, kernels, featureCounts):
        Kernel.__init__(self)
        self.m_n, self.m_norm, self.m_kernels, self.m_counts = len(kernels), 1.0/len(kernels), kernels, featureCounts

    def Eval(self, x1, x2=None):
        total, start = 0.0, 0.0
        for i in range(self.m_n):
            c = self.m_counts[i]
            if x2 is not None:
                total += self.m_norm * self.m_kenerls[i].Eval(x1[start:start+c], x2[start:start+c])
            else:
                total += self.m_norm * self.m_kenerls[i].Eval(x1[start:start+c])
            start+=c
        return total


