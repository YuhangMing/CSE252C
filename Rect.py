
# coding: utf-8

# In[1]:

class Rect:
    def __init__(self):
        self.m_xMin, self.m_yMin, self.m_width, self.m_height = 0, 0, 0, 0
    def __init__(self, xMin, yMin, width, height):
        self.m_xMin, self.m_yMin, self.m_width, self.m_height = xMin, yMin, width, height
    def __init__(self, rOther, isInt=False):
        if isInt:
            self.m_xMin, self.m_yMin, self.m_width, self.m_height = int(rOther.xMin), int(rOther.yMin), int(rOther.width), int(rOther.height)   
        else:
            self.m_xMin, self.m_yMin, self.m_width, self.m_height = rOther.xMin, rOther.yMin, rOther.width, rOther.height  
    def __str__(self):
        return '[origin:(%.2f,%.2f) size:(%.2f,%.2f)]' % (self.m_xMin, self.m_yMin, self.m_width, self.m_height)
    
    def Set(self,xMin, yMin, width, height):
        self.m_xMin, self.m_yMin, self.m_width, self.m_height = xMin, yMin, width, height
    def XMin(self):
        return self.m_xMin
    def SetXMin(self, xMin):
        self.m_xMin = xMin
    def YMin(self):
        return self.m_yMin
    def SetYMin(self, yMin):
        self.m_yMin = yMin
    def Width(self):
        return self.m_width
    def SetWidth(self, width):
        self.m_width = width
    def Height(self):
        return self.m_height
    def SetHeight(self, height):
        self.m_height = height
    def XMax(self):
        return self.m_xMin + self.m_width
    def YMax(self):
        return self.m_yMin + self.m_height
    def XCentre(self):
        return float(self.m_xMin) + float(self.m_width)/2.0
    def YCentre(self):
        return float(self.m_yMin) + float(self.m_height)/2.0
    def Area(self):
        return self.m_width*self.m_height
        
    def Translate(self, xtrans, ytrans):
        self.m_xMin, self.m_yMin = self.m_xMin + xtrans, self.m_yMin + ytrans
        
    def Overlap(self, rOther):
        x0 = max(self.XMin()+rOther.XMin())
        x1 = min(self.XMax()+rOther.XMax())
        y0 = max(self.YMin()+rOther.YMin())
        y1 = min(self.YMax()+rOther.YMax())
        if x0>=x1 and y0>=y1:
            return 0.0
        conArea = (x1-x0)*(y1-y0)
        return float(conArea) / float(self.Area() + rOther.Area() - conArea)
        
    def IsInside(self, rOther):
        return self.XMin()>=rOther.XMin() and self.YMin()>=rOther.YMin() and                 self.XMax()<=rOther.XMax() and self.YMax()<=rOther.YMax()
    
    
        

