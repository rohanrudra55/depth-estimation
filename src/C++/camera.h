#include<stdio.h>
#include<iostream>
// #include<Eigen/Dense>
#include<pcl/common/distances.h>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<opencv2/opencv.hpp>
#include<pcl/visualization/cloud_viewer.h>

struct camera{
    cv::Mat fundamental_matrix=cv::Mat::zeros(3,4,CV_64FC1);
    cv::Mat instrict_matrix;
    cv::Mat rotational_matrix;
    cv::Mat translation_matrix;
    double focal_length;
    double c_x;
    double c_y;
};
struct source{
    cv::Mat left_image;
    cv::Mat right_image;
    bool check;
    cv::Mat image_disparity;
    cv::Mat depth_map;
};
 std::ostream& operator<<(std::ostream& out,source& show){
    out<<"\r";
    cv::imshow("Left Image",show.left_image);
    cv::imshow("Right Image",show.right_image);
    cv::waitKey(0);
    return out;
}
std::ostream& operator<<(std::ostream& out,cv::Mat& data){
    cv::imshow("Data",data);
    cv::waitKey(0);
    out<<"\r";
    return out;
}