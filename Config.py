
# coding: utf-8

# In[1]:

from enum import Enum
import os

class Feature:
    def __init__(self):
        self.kernelName, self.featureName, self.params = '', '', []

class Config:
    #FeatureName = {'kFeatureTypeHaar':'haar', 'kFeatureTypeRaw':'raw', \
    #               'kFeatureTypeHistogram':'histogram'}
    #KernelName = {'kKernelLinear':'linear', 'kKernelTypeGaussian':'gaussian', \
    #             'kKernelTypeIntersection':'intersection', 'kKernelTypeChi2':'chi2'}
    def __init__(self):
        self.SetDefaults()
    def __init__(self, path):
        self.SetDefaults()       
        if os.path.isfile(path) is False:
            print 'Error: could not load config file' + path
            return
        f = open(path,'r')
        for line in f:
            words = filter(None, line.split())
            
            if (not words) or (len(words) < 2) or (words[0] is '#') or (words[1] is not '='):
                continue
            if words[0] is 'seed':
                self.seed = words[2]
            elif words[0] is 'quietMode':
                self.quietMode = words[2]
            elif words[0] is 'debugMode':
                self.debugMode = words[2]
            elif words[0] is 'sequenceBasePath':
                self.sequenceBasePath = words[2]
            elif words[0] is 'sequenceName':
                self.sequenceName = words[2]
            elif words[0] is 'resultsPath':
                self.resultsPath = words[2]
            elif words[0] is 'frameWidth':
                self.frameWidth = words[2]
            elif words[0] is 'frameHeight':
                self.frameHeight = words[2]
            elif words[0] is 'searchRadius':
                self.searchRadius = words[2]
            elif words[0] is 'svmC':
                self.svmC = words[2]
            elif words[0] is 'svmBudgetSize':
                self.svmBudgetSize = words[2]
            elif words[0] is 'feature':
                fkp = Feature()
                if len(words) < 4:
                    continue
                if words[2] is 'haar' or 'raw' or 'histogram':
                    fkp.featureName = words[2]
                else:
                    print 'Error: unrecognised feature: ' + words[2]
                    continue
                if words[3] is 'linear' or 'gaussian' or 'intersection' or 'chi2':
                    fkp.kernelName = words[3]
                    if words[3] is 'gaussian':
                        if len(words) is 5:
                            fkp.params.append(words[4])
                        else:
                            print 'Error: unreceived param. '  
                            continue
                else:
                    print 'Error: unrecognised kernel: ' + words[3]
                    continue
                self.features.append(fkp)
        
    def SetDefaults(self):
        self.quietMode, self.debugMode = False, False
        self.sequenceBasePath, self.sequenceName, self.resultsPath = '','',''
        self.frameWidth, self.frameHeight = 320, 240
        self.seed, self.searchRadius, self.svmC, self.svmBudgetSize = 0, 30, 1.0, 0
        self.features = []
        
    def __str__(self):
        print 'Config: '
        print '  QuietMode\t= ' + self.quieMode
        print '  DebugMode\t= ' + self.debugMode
        print '  SequenceBasePath\t= '+ self.sequenceBasePath
        print '  SequenceName\t= ' + self.sequenceName
        print '  ResultsPath\t= ' + self.resultsPath
        print '  FrameWidth\t= ' + self.frameWidth
        print '  FrameHeight\t= ' + self.frameHeight
        print '  Seed\t= ' + self.seed
        print '  SearchRadius\t= ' + self.searchRadius
        print '  svmC\t= ' + self.svmC
        print '  svmBudgetSize\t= ' + self.svmBudgetSize
        for i in range(self.features):
            print '  FeatureName\t= ' + self.features[i].featureName
            print '  KernelName\t= ' + self.features[i].kernelName
            for j in range(self.features[i].params):
                print '  ', self.features[i].params[j]
            print '\n'
        
    

