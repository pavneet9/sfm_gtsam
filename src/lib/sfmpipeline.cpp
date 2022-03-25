#include <sfmpipeline.hpp>
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>

using namespace std;
using namespace cv;
using namespace gtsam;


bool SfmPipeline::LoadImages(string file_path)
{

    std::string suffix = ".png";

    for(size_t i =10; i > 0 ; i--)
    {
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << i; // 0000, 0001, 0002, etc...
        std::string number = ss.str();
        std::string name = file_path + number + suffix;

        std::cout << name << '\n';

        cv::Mat img_grey = cv::imread(name, cv::IMREAD_GRAYSCALE);
        n_grey_images_.push_back(img_grey);
    }


    if(n_grey_images_.size() == 0)
    {
        std::cerr << "Something went bad, no files in the array" <<std::endl;
        return false;
    }
    return true;
}


/// Thing we need for bundle adjustment
// 1. Triangulated points for each image
// 2. Corresponsing keypoints to each traingulated image
// 3. Pose for each image 

bool SfmPipeline::RunPipeline(){

    // Get the Intrinsic Matrix for the camera used in the problem.
    GetIntrinsicMatrixes();
    
    // Calculate the sift features
    bool success;
    success = ComputeSift(); 

    pair<int, int> best_pair;
    success = FindBestPair(best_pair); 
    

    // We can make use of best pair to get essential matrix.
    // However in this case we will just go and find the Camera Poses for all subsequent camera frames
    success = GetCameraPose(); 
    
    bundle_adjustment();
    // Now we need to find the triangulated landmark point associated with matching keypoints

    return 1;

}



// This function returns the Intrinsic Matrix for the Camera. If it is not provided we can calculate it also
// Usually of the form
//  [ focal length    0          c_x
//    0            focal_length   c_y
//    0                0          1.0]
void SfmPipeline::GetIntrinsicMatrixes()
    {

    Mat K = (Mat_<double>(3,3) <<  1520.0,    0.0,      302.2,
                                    0.0,     1520.0,   246.87,
                                    0.0,      0.0,      1.0);

    Mat d = (Mat_<double>(1,5) << 0, 0, 0, 0, 0);
    
    camera_matrixes_.K = K;
    camera_matrixes_.d = d;
   }



// Calculate the descriptors and keypoints for each image using sift features.
bool SfmPipeline::ComputeSift(){
      int nfeatures=0;
      int nOctaveLayers=3;
      double contrastThreshold=0.04;
      double edgeThreshold=10;
      double sigma=1.6;
      cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create(nfeatures,nOctaveLayers,
                                                               contrastThreshold,edgeThreshold,sigma);
      
      for(size_t i =0; i < n_grey_images_.size(); i++)
      {
            vector<KeyPoint> keypoints;
            Mat descriptors;
            ImagePose img_pose;

            detector->detectAndCompute( n_grey_images_[i], Mat(), keypoints, descriptors );

            std::cout << "for image "<< i << "\n"
                    << "Number of SIFTs: " << descriptors.rows << "\n"
                    << "Size of each SIFT: " << descriptors.cols << "\n" ;
                    

            img_pose.keypoints = keypoints;
            img_pose.desc = descriptors;
            image_poses.emplace_back(img_pose);
      }
    return true;
}

bool SfmPipeline::FindBestPair(pair<int, int>& best_pair)
{     
      std::map<int, std::pair<int,int>,  std::greater<int>>  mapping_to_find_best_pair;       
      
      
      for(int query_image =0; query_image < n_grey_images_.size() -1; query_image++)
      { 

            auto &prev = image_poses[query_image];
            auto &cur = image_poses[query_image+1];
            pair<int,int> img_pair = std::make_pair(query_image, query_image + 1);  

            std::cout << "img_pair " << img_pair.first << img_pair.second << std::endl ;
            
            vector<DMatch> matches = KeypointMatcher(img_pair, prev, cur);

            std::cout << "size of the matches" << matches.size() << std::endl ;
            matches_between_pairs[img_pair] = matches; 
            mapping_to_find_best_pair[matches.size()] = img_pair;    

        
      }

    if (matches_between_pairs.size() > 0)
    {  
        best_pair = mapping_to_find_best_pair.begin()->second ;
        return true; 
    }
    else
    {
        return false;
    }

}



vector<DMatch> SfmPipeline::KeypointMatcher(const pair<int, int> &image_pair , ImagePose &img_pose_first, ImagePose &img_pose_second )
{
        
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    std::vector< std::vector<DMatch> > knn_matches;
    Mat descriptors_1 , descriptors_2;
    
    descriptors_1 = img_pose_first.desc;
    descriptors_2 = img_pose_second.desc;
    std::cout << "Number of SIFTs: " << descriptors_1.rows ;     

    matcher->knnMatch( descriptors_1, descriptors_2, knn_matches, 2);
    
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);

            // Store the matches in our Image Pose data structure, so we can later find correspondances easily
            img_pose_first.kp_match_idx(knn_matches[i][0].queryIdx, image_pair.second) =image_pair.first;
            img_pose_second.kp_match_idx(knn_matches[i][0].trainIdx , image_pair.first) =  image_pair.second;

        }
    }
 
    std::cout << "Number of Mathces: " << size(good_matches) << "\n";  
   
    return good_matches;
}


// This function is used to calculate the essential matrix between two images 
// We Find essential matrix between all subsequent images.
bool SfmPipeline::GetCameraPose()
{       
      std::cout << "total_pairs " << image_poses.size() << std::endl ;
      for(size_t i =0; i < image_poses.size() - 1; i++)
      { 
            std::cout << "img_pair " << i  << std::endl ;
            const pair<int, int> image_pair =  std::make_pair(i , i+1);

            auto &prev = image_poses[i];
            auto &cur = image_poses[i+1];

            if(i == 0)
            {
                 prev.camera_pose = Mat::eye(3, 4, CV_64F); 
                 prev.T = Mat::eye(4, 4, CV_64F);                    // First camera position is the origin 

            }
            vector<Point2d> src, dst;
            vector<size_t> kp_used;

            // We need the points in the image plane
            for (size_t k=0; k < prev.keypoints.size(); k++) {
                if (prev.kp_match_exist(k, i+1)) {                      // CHeck if a match actually exisits
                    size_t match_idx = prev.kp_match_idx(k, i+1);
                    src.push_back(prev.keypoints[k].pt);
                    dst.push_back(cur.keypoints[match_idx].pt);
                    kp_used.push_back(k);

                }
            }

        std::cout << kp_used.size() << "Match SIze" << std::endl ;
        if(kp_used.size() > 5) {
                cv::Mat mask;
                cv::Mat E = cv::findEssentialMat(src, dst,
                                                camera_matrixes_.K, RANSAC, 0.999, 1.0, mask); 

                std::cout << "Essential Matrix: " << E << "\n";
                        
                if(fabsf(determinant(E)) > 1e-07) {
                    
                    std::cout << "det(E) != 0 : " << determinant(E) << "\n";
            
                }
            
                Mat Rot, Translation;
                recoverPose(E,src, dst, camera_matrixes_.K ,
                                                Rot, Translation, mask);  
                
                std::cout << "R:\n" << Rot << std::endl;
                std::cout << "T:\n" << Translation << std::endl;

                if(fabsf(determinant(Rot))-1.0 > 1e-07) {
                        std::cerr<<"det(R) != +-1.0, this is not a rotation matrix"<<std::endl;
                }

            
            // local tansform
               Mat T = Mat::eye(4, 4, CV_64F);
               Rot.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
               Translation.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));
               std::cout << "Transformation:\n" << T << std::endl;

               // accumulate transform
               cur.camera_pose = T;

               cur.T = prev.T * T;
               std::cout << "Accumulated transformation:\n" << cur.T << std::endl;


                Get3DTriangulatedPoints(    
                                        image_pair,
                                        src,
                                        dst,
                                        kp_used
                                    );
        
        }

      }

     return true; 

}



bool SfmPipeline::Get3DTriangulatedPoints(    
                                    const pair<int, int> &image_pair,
                                    vector<Point2d> src,
                                    vector<Point2d> dst,
                                    vector<size_t> kp_used
 								)

{       
          int prev = image_pair.first;
          int cur = image_pair.second;

         // We have to keep track of which 2D points corredpoint to which 3D points
		  cv::Mat normalized_pt_img1, normalized_pt_img2;

		  cv::undistortPoints(src, normalized_pt_img1, camera_matrixes_.K, camera_matrixes_.d);
          cv::undistortPoints(dst, normalized_pt_img2, camera_matrixes_.K, camera_matrixes_.d);


          

         cv::Mat c1 = image_poses[prev].T.rowRange(0, 3);
         //camera_matrixes_.K.copyTo(c1.colRange(0, 3));
         cv::Mat c2 = image_poses[cur].T.rowRange(0, 3);

          //Find 3d points first in homogenous coordinates
		 cv::Mat pts3dHomogeneous;
         cv::triangulatePoints(c1, c2, normalized_pt_img1, normalized_pt_img2, pts3dHomogeneous);

          
          
          // convert to 3d 
		  cv::Mat pts3d;
		  cv::convertPointsFromHomogeneous(pts3dHomogeneous.t(),pts3d);
          
         
          cv::Mat rvec_cam1;
          cv::Rodrigues(c1.colRange(0, 3), rvec_cam1);
          std::cout << rvec_cam1 << std::endl;
          
          cv::Mat tvec_cam1(c1.colRange(3,4).t());

          vector<Point2d> projected_in_image1(src.size());
          cv::projectPoints(pts3d, rvec_cam1, tvec_cam1, camera_matrixes_.K, camera_matrixes_.d, projected_in_image1);

          cv::Mat rvec_cam2;
          cv::Rodrigues(c2.colRange(0, 3),rvec_cam2);
          cv::Mat tvec_cam2( c2.colRange(3, 4).t() );

          vector<Point2d> projected_in_image2(src.size());
          cv::projectPoints(pts3d, rvec_cam2, tvec_cam2, camera_matrixes_.K, camera_matrixes_.d, projected_in_image2);
          
          const float MIN_REPROJECTION_ERROR = 6.0;
          
          for(int i = 0; i < pts3d.rows; i++){

                //check if point reprojection error is small enough

                const float queryError = cv::norm(projected_in_image1[i]  - src[i]);
                const float trainError = cv::norm(projected_in_image2[i] - dst[i]);

                if(MIN_REPROJECTION_ERROR < queryError or
                    MIN_REPROJECTION_ERROR < trainError) continue;
                

                    CloudPoint p;
                    p.pt = cv::Point3d(pts3d.at<double>(i, 0),
                                        pts3d.at<double>(i, 1),
                                        pts3d.at<double>(i, 2));

                    size_t k = kp_used[i];
                    size_t match_idx = image_poses[prev].kp_match_idx(k, cur);

                    // Check if the landmark is already seen otherwise add it as a landmark
                    if (image_poses[prev].kp_3d_exist(k)) {
                        // Found a match with an existing landmark
                        image_poses[cur].kp_3d(match_idx) = image_poses[prev].kp_3d(k);
                        cummalative_point_cloud[image_poses[cur].kp_3d(match_idx)].seen++;
                    } 
                    else {
                        // Add new 3d point
                        CloudPoint landmark;

                        landmark.pt = p.pt;
                        landmark.seen = 2;

                        cummalative_point_cloud.push_back(landmark);  // list of landmarks

                        image_poses[prev].kp_3d(k) = cummalative_point_cloud.size() - 1;      // jst get the index of the last landmark
                        image_poses[cur].kp_3d(match_idx) = cummalative_point_cloud.size() - 1;
                    }
            }

	    return true;
	}


bool SfmPipeline::bundle_adjustment()
{      

        gtsam::Values result;
        double cx = 302.2;
        double cy = 246.87;
        double FOCAL_LENGTH = 1520.0;   
	    
        Cal3_S2 K(FOCAL_LENGTH, FOCAL_LENGTH, 0, cx, cy);
        noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

        NonlinearFactorGraph graph;
        Values initial;

        // Poses
        for (size_t i=0; i < image_poses.size(); i++) {
            auto &img_pose = image_poses[i];

            Rot3 R(
                img_pose.T.at<double>(0,0),
                img_pose.T.at<double>(0,1),
                img_pose.T.at<double>(0,2),

                img_pose.T.at<double>(1,0),
                img_pose.T.at<double>(1,1),
                img_pose.T.at<double>(1,2),

                img_pose.T.at<double>(2,0),
                img_pose.T.at<double>(2,1),
                img_pose.T.at<double>(2,2)
            );

            Point3 t;
            t(0) = img_pose.T.at<double>(0,3);
            t(1) = img_pose.T.at<double>(1,3);
            t(2) = img_pose.T.at<double>(2,3);

            Pose3 pose(R, t);

            // Add prior for the first image
            if (i == 0) {
                noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
                graph.emplace_shared<PriorFactor<Pose3> >(Symbol('x', 0), pose, pose_noise); // add directly to graph
            }

            initial.insert(Symbol('x', i), pose);
            // landmark seen
            // Then we add the image points into the graph
            for (size_t k=0; k < img_pose.keypoints.size(); k++) {
                if (img_pose.kp_3d_exist(k)) {
                    size_t landmark_id = img_pose.kp_3d(k);

                    if (cummalative_point_cloud[landmark_id].seen >= 2) {
                        Point2 pt;

                        pt(0) = img_pose.keypoints[k].pt.x;
                        pt(1) = img_pose.keypoints[k].pt.y;
                        std::cout << "landmark seen:\n" << landmark_id << std::endl; 
                        std::cout << "image:\n" << i << std::endl; 

                        graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(pt, measurement_noise, Symbol('x', i), Symbol('l', landmark_id), Symbol('K', 0));
                    }
                }
            }
        }

        // Add a prior on the calibration.
        initial.insert(Symbol('K', 0), K);

        noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 100, 100, 0.01 , 100, 100).finished());
        graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);

        // Initialize estimate for landmarks
        bool init_prior = false;

        for (size_t i=0; i < cummalative_point_cloud.size(); i++) {
            if (cummalative_point_cloud[i].seen >= 2) {
                cv::Point3d &p = cummalative_point_cloud[i].pt;
                //std::cout << "landmark added:\n" << i << std::endl; 

                initial.insert<Point3>(Symbol('l', i), Point3(p.x, p.y, p.z));

                if (!init_prior) {
                    init_prior = true;

                    noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, 0.1);
                    Point3 p(cummalative_point_cloud[i].pt.x, cummalative_point_cloud[i].pt.y, cummalative_point_cloud[i].pt.z);
                    graph.emplace_shared<PriorFactor<Point3>>(Symbol('l', i), p, point_noise);
                }

            }
        }

        result = LevenbergMarquardtOptimizer(graph, initial).optimize();

        cout << endl;
        cout << "initial graph error = " << graph.error(initial) << endl;
        cout << "final graph error = " << graph.error(result) << endl;
        result.print("Final results:\n");
        return true;
}



	




/******************************* HELPER FUNCTIONS **************************************************/


bool SfmPipeline::CheckCoherentRotation(cv::Mat& R){

   if(fabsf(determinante(R))-1.0 > 1e-07) {

      std::cout << "det(R) != +-1.0, this is not a rotation matrix" << std::endl;
      return false;
    }
    return true;
}



void SfmPipeline::MergePointClouds(const std::vector<CloudPoint>& new_point_cloud) {

  std::cout << "Adding new points..." << new_point_cloud.size() << std::endl;

  const float MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE   = 0.01;
//  const float MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE = 20.0;

      size_t newPoints = 0;
 //     size_t mergedPoints = 0;

      for(const CloudPoint& p : new_point_cloud) {
          const cv::Point3d new_point = p.pt; //new 3D point

          bool found_matching_3d_point = false;
          for(CloudPoint& existing_point : cummalative_point_cloud) {
              if(cv::norm(existing_point.pt - new_point) < MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE){
                  //This point is very close to an existing 3D cloud point
                  found_matching_3d_point = true;
                  break;

               }
          }

          if( not found_matching_3d_point) {
              //This point did not match any existing cloud points - add it as new.
              cummalative_point_cloud.push_back(p);
              newPoints++;
          }
      }

      std::cout << "New points:" << newPoints << std::endl;
}


double SfmPipeline::determinante(cv::Mat& relativeRotationCam){

  Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,
                                  Eigen::Dynamic,
                                  Eigen::RowMajor> > eigenMatrix((double *)relativeRotationCam.data,3,3);

  Eigen::FullPivLU<Eigen::Matrix<double, Eigen::Dynamic,
                                         Eigen::Dynamic,
                                         Eigen::RowMajor>> eigenMatrixV2(eigenMatrix);

  double det = eigenMatrixV2.determinant();
  return det;
}