# coding: utf-8
class Feature:

    def __init__(self):
        self.kernelName, self.featureName, self.params = '', '', []


class Config:
    # FeatureName = {'kFeatureTypeHaar':'haar', 'kFeatureTypeRaw':'raw',
    #               'kFeatureTypeH==togram':'h==togram'}
    # KernelName = {'kKernelLinear':'linear', 'kKernelTypeGaussian':'gaussian',
    #             'kKernelTypeIntersection':'intersection', 'kKernelTypeChi2':'
    #              chi2'}

    def __init__(self, path=None):
        self.setdefaults()
        if path is None:
            return
        try:
            f = open(path, 'r')
            for line in f:
                try:
                    words = filter(None, line.split())
                except ValueError:
                    print 'Error: can not accurately read config file'
                else:
                    if (not words) or (len(words) < 2) or (words[0] == '#') or (words[1] != '='):
                        continue
                    if words[0] == 'seed':
                        self.seed = int(words[2])
                    elif words[0] == 'quietMode':
                        self.quietMode = words[2]
                    elif words[0] == 'debugMode':
                        self.debugMode = words[2]
                    elif words[0] == 'sequenceBasePath':
                        self.sequenceBasePath = words[2]
                    elif words[0] == 'sequenceName':
                        self.sequenceName = words[2]
                    elif words[0] == 'resultsPath':
                        self.resultsPath = words[2]
                    elif words[0] == 'frameWidth':
                        self.frameWidth = int(words[2])
                    elif words[0] == 'frameHeight':
                        self.frameHeight = int(words[2])
                    elif words[0] == 'searchRadius':
                        self.searchRadius = int(words[2])
                    elif words[0] == 'svmC':
                        self.svmC = float(words[2])
                    elif words[0] == 'svmBudgetSize':
                        self.svmBudgetSize = int(words[2])
                    elif words[0] == 'feature':
                        fkp = Feature()
                        if len(words) < 4:
                            continue
                        if words[2] == 'haar' or words[2] == 'raw' or words[2] == 'histogram':
                            fkp.featureName = words[2]
                        else:
                            print 'Error: unrecognised feature: ' + words[2]
                            continue
                        if words[3] == 'linear' or words[3] == 'gaussian' or words[3] == 'intersection' or words[3] == 'chi2':
                            fkp.kernelName = words[3]
                            if words[3] == 'gaussian':
                                if len(words) == 5:
                                    fkp.params.append(words[4])
                                else:
                                    print 'Error: unreceived param. '
                                    continue
                        else:
                            print 'Error: unrecognised kernel: ' + words[3]
                            continue
                        self.features.append(fkp)
        except IOError as err:
            print 'File error: ' + str(err)

    def setdefaults(self):
        self.quietMode, self.debugMode = False, False
        self.sequenceBasePath, self.sequenceName, self.resultsPath = '', '', ''
        self.frameWidth, self.frameHeight = 320, 240
        self.seed, self.searchRadius, self.svmC, self.svmBudgetSize = 0, 30, 1.0, 0
        self.features = []

    def printstr(self):
        print 'Config: '
        print '  QuietMode\t= %s' % self.quietMode
        print '  DebugMode\t= %s' % self.debugMode
        print '  SequenceBasePath\t= %s' % self.sequenceBasePath
        print '  SequenceName\t= %s' % self.sequenceName
        print '  ResultsPath\t= %s' % self.resultsPath
        print '  FrameWidth\t= %.2f' % self.frameWidth
        print '  FrameHeight\t= %.2f' % self.frameHeight
        print '  Seed\t= %.2f' % self.seed
        print '  SearchRadius\t= %.2f' % self.searchRadius
        print '  svmC\t= %.2f' % self.svmC
        print '  svmBudgetSize\t= %.2f' % self.svmBudgetSize
        for i in range(len(self.features)):
            print '  FeatureName\t= %s' % self.features[i].featureName
            print '  KernelName\t= %s' % self.features[i].kernelName
            for j in range(len(self.features[i].params)):
                print '  ', self.features[i].params[j]
            print '\n'
