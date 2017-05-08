//
//  IntegImg.cpp
//  Struck
//
//  Created by Yuhang Ming on 07/05/2017.
//  Copyright © 2017 Yuhang Ming. All rights reserved.
//

#include "IntegImg.hpp"
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

IntegImg::IntegImg(const Mat& img, bool color){
    
    channel = color? 3 : 1;
    
    // initiate the size of image vector
    for (int i=0; i<channel; i++){
        
        // original image
        imgs.push_back(Mat(img.rows, img.cols, CV_8UC1));
        
        // integral image
        integ_imgs.push_back(Mat(img.rows, img.cols, CV_32SC1));
    }
    
    // put original images into image vectors
    if (color){
        split(img, imgs);
    } else {
        
        if(img.channels() == 3){
            cvtColor(img, imgs[0], CV_RGB2GRAY);
        }
        
        else if(img.channels() == 1){
            img.copyTo(imgs[0]);
        }
        
    }
    
    // compute the integral image
    for (int i=0; i<channel; i++){
        // every pixel is the integral of left and up image
        integral(imgs[i], integ_imgs[i]);
    }
    
}


int IntegImg::CalSum(const FloatRect& rect){
    
    int row = imgs[0].rows;
    int col = imgs[0].cols;
//    cout << integ_imgs.size() << endl;
    Mat my_int_img = integ_imgs[channel-1];
    
    int x_min = int(col * rect.getX());
    int y_min = int(row * rect.getY());
    int x_max = x_min + int(col * rect.getWidth());
    int y_max = y_min + int(row * rect.getHeight());
    
//    cout << x_min << " " << y_min << " " << x_max << " " << y_max<<endl;

    int A = my_int_img.at<int>(y_min, x_min);
    int B = my_int_img.at<int>(y_max, x_min);
    int C = my_int_img.at<int>(y_min, x_max);
    int D = my_int_img.at<int>(y_max, x_max);
    
    return D + A - B - C;
}


