#!/usr/bin/env python
class Rect:
    def __init__(self, x=0, y=0, width=0, height=0):
        self.set(x, y, width, height)

    def initFromRect(self, r):
        self.set(r.getX(), r.getY(), r.getWidth(), r.getHeight())

  
    # Getters
    def getX(self):
        return self.x

    def getXCenter(self):
        return self.x + self.width /2

    def getXMax(self):
        return self.x + self.width

    def getY(self):
        return self.y

    def getYCenter(self):
        return self.y + self.height /2

    def getYMax(self):
        return self.y + self.height

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getArea(self):
        return self.width * self.height

     
    # Setters
    def set(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def setWidth(self, w):
        self.width = x

    def setHeight(self, h):
        self.height = h


    # Other useful methods
    def isInside(self, r2):
        xIn = (r2.getX() <= self.getX()) and (self.getXMax() <= r2.getXMax())
        yIn = (r2.getY() <= self.getY()) and (self.getYMax() <= r2.getYMax())
        return (xIn and yIn)

    def overlap(self, r2):
        x = max(self.getX(), r2.getX())
        y = max(self.getY(), r2.getY())
        xMax = min(self.getXMax(), r2.getXMax())
        yMax = min(self.getYMax(), r2.getYMax())

        width = xMax - x
        height = yMax - y
        
        if( width <= 0 or height <= 0 ):
            return 0
        
        intersection = width * height
        union = self.getArea() + r2.getArea() - intersection
        return intersection / union

    def translate(self, dx, dy):
        self.x += dx
        self.y += dy

###################### Debug Methods ###########################
    # Debug getters
    def XMin(self):
        return self.x

    def XCentre(self):
        return self.x + self.width/2

    def XMax(self):
        return self.x + self.width

    def YMin(self):
        return self.y

    def YCentre(self):
        return self.y + self.height/2

    def YMax(self):
        return self.y + self.height

    def Width(self):
        return self.width

    def Height(self):
        return self.height

    def Area(self):
        return self.width * self.height

    # Setters
    def Set(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def SetXMin(self, x):
        self.x = x

    def SetYMin(self, y):
        self.y = y

    def SetWidth(self, w):
        self.width = x

    def SetHeight(self, h):
        self.height = h

    
    # Other useful methods
    def IsInside(self, r2):
        xIn = (r2.getX() <= self.getX()) and (self.getXMax() <= r2.getXMax())
        yIn = (r2.getY() <= self.getY()) and (self.getYMax() <= r2.getYMax())
        return (xIn and yIn)

    def Overlap(self, r2):
        x = max(self.getX(), r2.getX())
        y = max(self.getY(), r2.getY())
        xMax = min(self.getXMax(), r2.getXMax())
        yMax = min(self.getYMax(), r2.getYMax())

        width = xMax - x
        height = yMax - y
        
        if( width <= 0 or height <= 0 ):
            return 0
        
        intersection = width * height
        union = self.getArea() + r2.getArea() - intersection
        return intersection / union

    def Translate(self, dx, dy):
        self.x += dx
        self.y += dy
