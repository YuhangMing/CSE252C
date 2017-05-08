//
//  HaarFeature.hpp
//  Struck
//
//  Created by Yuhang Ming on 07/05/2017.
//  Copyright © 2017 Yuhang Ming. All rights reserved.
//

#ifndef HaarFeature_hpp
#define HaarFeature_hpp

#include <stdio.h>
#include <vector>

#include "Rect.hpp"
#include "IntegImg.hpp"

class HaarFeature{
    std::vector<FloatRect> rects;
    std::vector<int> weights;

public:
    HaarFeature(const FloatRect& rect, int idt);
    int Eval(IntegImg integ_img);
};

#endif /* HaarFeature_hpp */
