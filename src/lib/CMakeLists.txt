add_library(sfm_pipeline sfmpipeline.cpp)
target_link_libraries(sfm_pipeline ${OpenCV_LIBS})

add_library(visualizer visualizer_pcl.cpp)
 target_link_libraries(visualizer ${OpenCV_LIBS} ${PCL_LIBRARIES} pcl_visualization )

