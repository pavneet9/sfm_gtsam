#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/imgcodecs.hpp>

#include <opencv2/highgui.hpp>
#include <vector>
#include <map>
#include <cassert>

#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace cv;

struct CameraMatrixes
{   
    cv::Matx33d K;
    cv::Mat d;

};

struct CloudPoint {
	cv::Point3d pt;
  int seen;
};

struct ImagePose
    {  
        cv::Mat camera_pose; // The pose of the camera
        cv::Mat T; // 4x4 pose transformation matrix

        cv::Mat desc; // feature descriptor
        std::vector<cv::KeyPoint> keypoints; // keypoint

        // alias to clarify map usage below
        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        std::map<kp_idx_t, std::map<img_idx_t, kp_idx_t>> kp_matches; // keypoint matches in other images
        std::map<kp_idx_t, landmark_idx_t> kp_landmark; // seypoint to 3d points

        // helper
        kp_idx_t& kp_match_idx(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx][img_idx]; };
        bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };

        landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; }
        bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; }
   };


class SfmPipeline
{
    private:
      std::set<int>                           n_done_views_;
      std::set<int>                           n_good_views_;
      std::vector<std::vector<cv::KeyPoint>>  images_keypoints_;
      std::vector<cv::Mat>                    images_descriptors_;
      std::vector<std::vector<cv::Point2d>>   imagesPts2D;     
      std::map<std::pair<int,int>, std::vector<DMatch>>  matches_between_pairs;       
      std::vector<std::pair<int,int>>          pairs_with_bad_pnp;
    
    public:
       std::vector<ImagePose>                  image_poses;
       std::vector<cv::Mat>                    n_grey_images_;
       std::vector<CloudPoint>                 cummalative_point_cloud;

       std::vector<cv::Point3d>                points_3d;
       std::map<int, cv::Matx34d>              camera_poses;
       std::vector<int>                        views_for_pose;
       std::vector<int>                        views_for_triangulation;
       std::vector<CloudPoint>                 cummalative_point_cloud1;
       CameraMatrixes                          camera_matrixes_;
       int                                     origin_image;


    bool LoadImages(std::string file_path);
    
    bool RunPipeline();

    void GetIntrinsicMatrixes();

    bool FindBestPair(std::pair<int, int> &best_pair);

    bool ComputeSift();

    bool GetCameraPose( );

    vector<DMatch> KeypointMatcher(const pair<int, int> &image_pair, ImagePose &img_pose_first, ImagePose &img_pose_second );

    int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);

    void symmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
                                    const std::vector<std::vector<cv::DMatch> >& matches2,
                                    std::vector<cv::DMatch>& symMatches );

    bool Get3DTriangulatedPoints(    
                                    const pair<int, int> &image_pair,
                                    vector<Point2d> src,
                                    vector<Point2d> dst,
                                    vector<size_t> kp_used
 								);

    bool AddNewViews();

    bool Get2dto3dMatches(  
                            int view,
                            std::vector<cv::Point2d>& points_2d,
                            std::vector<cv::Point3d>& points_3d,
                            int& best_view,
                            std::pair<int, int>& img_pair 
                       );

    bool FindCameraPosePNP(
                        cv::Matx34d camera_pose 
                      , std::vector<cv::Point3d> points_3d  
                      , std::vector<cv::Point2d> points_2d
                      , cv::Matx34d new_camera_pose
                      );

    void MergePointClouds(const std::vector<CloudPoint>& new_point_cloud);


    bool FindBestMatch(
                                int view,
                                int& best_view,
                                std::vector<cv::DMatch>& best_matches
                        );
    
    bool CheckCoherentRotation(cv::Mat& R);

    double determinante(cv::Mat& relativeRotationCam);

    bool bundle_adjustment();
};

