//
//  Rect.hpp
//  Struck
//
//  Created by Yuhang Ming on 07/05/2017.
//  Copyright © 2017 Yuhang Ming. All rights reserved.
//

#ifndef Rect_hpp
#define Rect_hpp

#include <stdio.h>
#include <iostream>
#include <algorithm>

template<typename T>
class Rect{
    
    T x_min, y_min, width, height;
    
public:
    
    Rect(T x=0, T y=0, T w=0, T h=0){
        x_min = x;
        y_min = y;
        width = w;
        height = h;
    }
    
    T getX() const { return x_min; }
    T getY() const { return y_min; }
    T getWidth() const { return width; }
    T getHeight() const { return height; }
    
};

typedef Rect<float> FloatRect;



#endif /* Rect_hpp */
