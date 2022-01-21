#include"camera.h"

//Funcations
void depth_to_pointcloud(source& map,cv::Mat& left_image,pcl::PointCloud<pcl::PointXYZRGB> &point_cloud, float max_depth,camera& left_camera) {

    point_cloud.height = (uint32_t) map.depth_map.rows;
    point_cloud.width  = (uint32_t) map.depth_map.cols;
    point_cloud.is_dense = false;
    point_cloud.resize(point_cloud.height * point_cloud.width);

    for (int h = 0; h < (int) map.depth_map.rows; h++) {
        for (int w = 0; w < (int) map.depth_map.cols; w++) {

            pcl::PointXYZRGB &pt = point_cloud.at(h * point_cloud.width + w);
            

            switch (left_image.channels()) {
                case 1: {
                    unsigned char v = left_image.at<unsigned char>(h, w);
                    pt.b = v;
                    pt.g = v;
                    pt.r = v;
                }
                    break;
                case 3: {
                    cv::Vec3b v = left_image.at<cv::Vec3b>(h, w);
                    pt.b = v[0];
                    pt.g = v[1];
                    pt.r = v[2];
                }
                    break;
                }

            float depth = 0.f;
            switch (map.depth_map.type()) {
                case CV_16UC1: // unit is mm
                    depth = float(map.depth_map.at<unsigned short>(h, w));
                    depth *= 0.001f; // convert to meter for pointcloud
                    break;
                case CV_32FC1: // unit is meter
                    depth = map.depth_map.at<float>(h, w);
                    break;
            }

            if (std::isfinite(depth) && depth >= 0 && depth < max_depth) {
                double W = depth / left_camera.focal_length; 
                pt.x = float((cv::Point2f(w, h).x - left_camera.c_x) * W);
                pt.y = float((cv::Point2f(w, h).y - left_camera.c_y) * W);
                pt.z = depth;
            } else
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
        }
    }
}
void read_parameters(camera& data,char side){
    // Reading lines form the txt file
    FILE *line;
    if(side=='l')
        line=fopen("../resource/left/000000.txt","r");
    else
        line=fopen("../resource/right/000000.txt","r");
    // If the file is not read then the code will stop here to prevent errors in further calculations
    if(line==NULL){
        puts("File not found!\n");
        exit(0);
    }
    // reading each number from the file and storing it in the matrix
    int colum=0,row=0;
    double number;
    while(fscanf(line,"%lf",&number)!=EOF){
        if(colum==4){
            colum=0;
            row++;
        }
        data.fundamental_matrix.at<double>(row,colum)=number;
        colum++;
        if(row==4 && colum==2)
            break;
    }
    cv::decomposeProjectionMatrix(data.fundamental_matrix,data.instrict_matrix,data.rotational_matrix,data.translation_matrix);
    data.c_x=data.instrict_matrix.at<double>(0,2);
    data.c_y=data.instrict_matrix.at<double>(1,2);
}
cv::Mat pre_processing(cv::Mat& image){
    cv::Mat smoothen_image;
    // Applying gaussian blur to remove some noise
    cv::GaussianBlur(image, smoothen_image, cv::Size(5, 5), 0.1);
    // Equilizing the histograms to fix excessive contrast and exposures
    cv::Mat histogram_equilized;
    // Normal method to equilize histogrms
    // equalizeHist(gamma_corrected,histogram_equilized);
    // Using a adaptive method to equilize histogram
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->apply(smoothen_image,histogram_equilized);
    // Doing gamma correcting to control shadows and highlights
    float gamma_=1.5;
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);
    cv::Mat gamma_corrected = histogram_equilized.clone();
    LUT(histogram_equilized, lookUpTable, gamma_corrected);

    // To store all the filtered images addresss 
    cv::Mat *filtered_images[]={&histogram_equilized,&gamma_corrected,&smoothen_image};

    return *filtered_images[1];
}
void disparity_calculate(source& processed,camera& left,camera& right){
    // To creat the disparity between left and right image
    // Stereo Binary SGBM is a better block matching then BM
    cv::Ptr<cv::StereoSGBM>sgbm = cv::StereoSGBM::create(0,16*6,15,4000,15000,0,0,0,0,0,cv::StereoSGBM::MODE_SGBM_3WAY);
    cv::Mat disparity_map;
    sgbm->compute(processed.left_image, processed.right_image, disparity_map);   
    //Changing the format of disparity map to make it more readble    
    disparity_map.convertTo(processed.image_disparity,CV_32FC1, 1 / 16.f); 
    // camera left,right;
    read_parameters(left,'l');
    read_parameters(right,'r');

    left.focal_length=left.instrict_matrix.at<double>(0,0);
    double baseline=right.translation_matrix.at<double>(0)-left.translation_matrix.at<double>(0);
    processed.depth_map=cv::Mat::zeros(disparity_map.size(),disparity_map.type());
    // double lcx=left.instrict_matrix.at<double>(0,2);
    // double rcx=left.instrict_matrix.at<double>(0,2);
    // double diff=rcx-lcx;
    std::cout<<disparity_map;
    for(int _row=0;_row<(int)disparity_map.rows;_row++){
        for(int _col=0;_col<(int)disparity_map.cols;_col++){
            double pixel=disparity_map.at<double>(_row,_col);
            double depth=0.1;
            if(pixel>0 && baseline>0 && left.focal_length>0)
                depth = (left.focal_length*baseline)/pixel;
            processed.depth_map.at<double>(_row,_col)=depth;
            // std::cout<<depth;
        }
    }
    // Scaling and color changeing
    // double min, max;
    // cv::minMaxLoc(depth_map, &min, &max);

    // cv::Mat mat_scaled,color_map;
    // if (min != max)
    //     depth_map.convertTo(mat_scaled, CV_8UC1, 255.0 / (max - min), 0);
    // cv::applyColorMap(mat_scaled, color_map, int());
    // std::cout<<color_map;
}

int main(int argc,char* argv[]){
    source raw;
    if(argc>1){
        std::string file_name=argv[1];
        raw.left_image=cv::imread("../resource/left/"+file_name,cv::IMREAD_GRAYSCALE);
        raw.right_image=cv::imread("../resource/right/"+file_name,cv::IMREAD_GRAYSCALE); 
    }
    else{
        raw.left_image=cv::imread("../resource/left/000004.png",cv::IMREAD_GRAYSCALE);
        raw.right_image=cv::imread("../resource/right/000004.png",cv::IMREAD_GRAYSCALE);
    }
    //Dsipalying the read data
    // std::cout<<raw;
    source processed;
    camera left,right;
    // Processing left image
    processed.left_image=(pre_processing(raw.left_image));
    // Processimng the right image
    processed.right_image=(pre_processing(raw.right_image));
    std::cout<<processed;
    // Calcuting desparity map
    disparity_calculate(processed,left,right);
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    depth_to_pointcloud(processed,raw.left_image,cloud,5.f,left);
    pcl::io::savePCDFileASCII("Test_pcd.pcd",cloud);
    // Showing disparity map
    // std::cout<<processed.image_disparity;
}