//
//  IntegImg.hpp
//  Struck
//
//  Created by Yuhang Ming on 07/05/2017.
//  Copyright © 2017 Yuhang Ming. All rights reserved.
//

#ifndef IntegImg_hpp
#define IntegImg_hpp

#include <stdio.h>

#include "Rect.hpp"

#include <opencv/cv.h>
#include <eigen3/Eigen/Core>
#include <vector>

class IntegImg{
    int channel;
    std::vector<cv::Mat> imgs, integ_imgs;
    
public:
    IntegImg(const cv::Mat& img, bool color);
    int CalSum(const FloatRect& rect);
};


#endif /* IntegImg_hpp */
