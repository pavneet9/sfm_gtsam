#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/xfeatures2d.hpp>
#include "opencv2/features2d.hpp"

#include <opencv2/imgcodecs.hpp>

#include <lib/sfmpipeline.hpp>


int make_pcl_visualization(std::vector<CloudPoint> pointcloud);

void SORFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);