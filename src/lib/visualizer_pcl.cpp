#include "visualizer_pcl.hpp"
#include <vector>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/io.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/xfeatures2d.hpp>
#include "opencv2/features2d.hpp"

#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv; 


void SORFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	
	std::cerr << "Cloud before SOR filtering: " << cloud->width * cloud->height << " data points" << std::endl;
	

	// Create the filtering object
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud (cloud);
	sor.setMeanK (50);
	sor.setStddevMulThresh (1.0);
	sor.filter (*cloud_filtered);
	
	std::cerr << "Cloud after SOR filtering: " << cloud_filtered->width * cloud_filtered->height << " data points " << std::endl;
	
	copyPointCloud(*cloud_filtered,*cloud);
	

}	



int make_pcl_visualization(vector<CloudPoint> pointcloud)
{

          using CloudType = pcl::PointCloud<pcl::PointXYZ>;
          CloudType::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        
            for(size_t i = 0; i < pointcloud.size(); i++)
            {   
                    Point3d pt_3d = pointcloud[i].pt;

                    CloudType::PointType p;
                    p.x = pt_3d.x;
                    p.y = pt_3d.y;
                    p.z = pt_3d.z;

                    cloud->push_back(p);
            }
            
            pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);
            
            std::cout << "Loaded "
                        << cloud->width * cloud->height
                        << " data points from test_pcd.pcd with the following fields: "
                        << std::endl;

            SORFilter(cloud);

   
             pcl::visualization::CloudViewer viewer("Cloud Viewer");
             cloud->is_dense = false;    
    //blocks until the cloud is actually rendered
            viewer.showCloud(cloud);

            //pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
            //viewer.showCloud(cloud);

            while (!viewer.wasStopped())
            {
            }


 // pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
 // viewer->setBackgroundColor (0, 0, 0);
 //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud1, 0, 255, 0);
 // viewer->addPointCloud<pcl::PointXYZ> (cloud1, single_color, "sample cloud");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //->addCoordinateSystem (1.0);
//  viewer->initCameraParameters ();



            return (0);

}