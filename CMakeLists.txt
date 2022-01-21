cmake_minimum_required(VERSION 2.8.12)
project(stereo)
find_package(OpenCV REQUIRED)
find_package(PCL 1.2 REQUIRED)
file(GLOB_RECURSE VTK_LIB "/usr/local/lib/libopencv_viz.so")
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable( stereo src/C++/Stereo_Depth_Estimation.cpp )
target_link_libraries( stereo ${OpenCV_LIBS} ${VTK_LIB} ${PCL_LIBRARIES} )
message(
    "
                                             @@@        @   @@@                                 
                                              @@@      @@@ @@@                                  
                                               (@@    @@@@@@@                                   
                                                 @@  @@@ @@@                                    
                                                  @@@@@ @@@@@                                   
                                                   @@@ @@@ @@@                                  
                                                  @@@@@@@   @@@                                 
                                                 @@@ @@%     *@@                                


                                     @@@  @ @   @  @ @@@    @        %@    @                    
                                      @   &  @ @   @ @ @   @ @      @& @   @                    
                                      @   &   @    @   @  @   @    @,   @  @                    

                                                  TiVRA AI     
    "
)
