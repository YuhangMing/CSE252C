//
//  HaarFeature.cpp
//  Struck
//
//  Created by Yuhang Ming on 07/05/2017.
//  Copyright © 2017 Yuhang Ming. All rights reserved.
//

#include "HaarFeature.hpp"


HaarFeature::HaarFeature(const FloatRect& rect, int idt){
    
    switch (idt) {
        case 0:
            // horizontal edge feature
            rects.push_back(FloatRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight()/2));
            rects.push_back(FloatRect(rect.getX(), rect.getY()+rect.getHeight()/2, rect.getWidth(), rect.getHeight()/2));
            weights.push_back(1);
            weights.push_back(-1);
            break;
            
        case 1:
            // vertical edge feature
            rects.push_back(FloatRect(rect.getX(), rect.getY(), rect.getWidth()/2, rect.getHeight()));
            rects.push_back(FloatRect(rect.getX()+rect.getWidth()/2, rect.getY(), rect.getWidth()/2, rect.getHeight()));
            weights.push_back(1);
            weights.push_back(-1);
            break;
        
        case 2:
            // horizontal line feature
            rects.push_back(FloatRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight()/3));
            rects.push_back(FloatRect(rect.getX(), rect.getY()+rect.getHeight()/3, rect.getWidth(), rect.getHeight()/3));
            rects.push_back(FloatRect(rect.getX(), rect.getY()+rect.getHeight()*2/3, rect.getWidth(), rect.getHeight()/3));
            weights.push_back(1);
            weights.push_back(-2);
            weights.push_back(1);
            break;
        
        case 3:
            // vertical line feature
            rects.push_back(FloatRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight()/3));
            rects.push_back(FloatRect(rect.getX()+rect.getWidth()/3, rect.getY(), rect.getWidth()/3, rect.getHeight()));
            rects.push_back(FloatRect(rect.getX()+rect.getWidth()*2/3, rect.getY(), rect.getWidth()/3, rect.getHeight()));
            weights.push_back(1);
            weights.push_back(-2);
            weights.push_back(1);
            break;
        
        case 4:
            // diagonal feature
            rects.push_back(FloatRect(rect.getX(), rect.getY(), rect.getWidth()/2, rect.getHeight()/2));
            rects.push_back(FloatRect(rect.getX()+rect.getWidth()/2, rect.getY(), rect.getWidth()/2, rect.getHeight()/2));
            rects.push_back(FloatRect(rect.getX(), rect.getY()+rect.getHeight()/2, rect.getWidth()/2, rect.getHeight()/2));
            rects.push_back(FloatRect(rect.getX()+rect.getWidth()/2, rect.getY()+rect.getHeight()/2, rect.getWidth()/2, rect.getHeight()/2));
            weights.push_back(1);
            weights.push_back(-1);
            weights.push_back(1);
            weights.push_back(-1);
            break;
        
        case 5:
            // center surround feature
            rects.push_back(FloatRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight()));
            rects.push_back(FloatRect(rect.getX()+rect.getWidth()/4, rect.getY()+rect.getHeight()/4, rect.getWidth()/2, rect.getHeight()/2));
            weights.push_back(1);
            weights.push_back(-4);
            break;
    }
    
}

int HaarFeature::Eval(IntegImg integ_img){
    int num_rect = int(rects.size());
    int value = 0;
    
    for(int i = 0; i < num_rect; i++){
        value += weights[i] * integ_img.CalSum(rects[i]);
    }
    
    return value;
}
