//
//  main.cpp
//  Struck
//
//  Created by Yuhang Ming on 26/04/2017.
//  Copyright © 2017 Yuhang Ming. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <string>


#include "Rect.hpp"
#include "IntegImg.hpp"
#include "HaarFeature.hpp"


using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    
//    load image
    Mat img = imread("/Users/Yohann/Documents/Xcode_C++/0001.jpg",1);
//    Mat img = imread("/Users/Yohann/Documents/Xcode_C++/img.jpg",1);
    if(!img.data){
        printf("No image data \n");
    }
    
//    //display image
//    namedWindow("Display Image", CV_WINDOW_AUTOSIZE);
//    imshow("Display Image", img);
//    namedWindow("Original Frame", WINDOW_AUTOSIZE);
//    
//    waitKey(0);
    
//    int rows = img.rows;
//    int cols = img.cols;
//    cout << rows << "   " << cols << endl;
    
    // calculate integral image
    // an image with each pixel value equals to the integral
    bool color = false;
    IntegImg Int_Img(img, color);
    
    
    // 4x4 grid, 2 scales, 6 types
//    static const int feature_num = 192;
    vector<HaarFeature> haar_features;
    vector<float> haar_feature_val;
    float x[] = {0, 0.25, 0.5, 0.75};
    float y[] = {0, 0.25, 0.5, 0.75};
    float s[] = {0.125, 0.25};
    
    // 4x4 grid
    for(int idx = 0; idx < 4; idx++){
        for(int idy = 0; idy < 4; idy++){
            // 2 scales
            for(int ids = 0; ids < 2; ids++){
                // float rectangle
                FloatRect rect(x[idx], y[idy], s[ids], s[ids]);
                // 6 types
                for(int idt = 0; idt < 6; idt++){
                    
                    HaarFeature my_feature = HaarFeature(rect, idt);
                    haar_features.push_back(my_feature);
                    haar_feature_val.push_back(my_feature.Eval(Int_Img));
//                    int pixel_sum = Int_Img.CalSum(rect);
                    
                }
            }
        }
    }
    
    // display the feature values
    for(int i = 0; i < 192; i++){
        cout << haar_feature_val[i] << "    ";
    }
    cout << endl;
    // 2 scales on the 4x4 grid
    
    
    return 0;
}
