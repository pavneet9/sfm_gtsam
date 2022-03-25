
#include <string>
#include <vector>
#include <lib/sfmpipeline.hpp>
#include <lib/visualizer_pcl.hpp>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using std::string;
using std::vector;

using namespace cv;
using namespace cv::xfeatures2d;




int main(){

    SfmPipeline pipeline = SfmPipeline();

    pipeline.LoadImages("../../img/temple/temple");
    pipeline.RunPipeline();    


    make_pcl_visualization(pipeline.cummalative_point_cloud);

    return 0;

}