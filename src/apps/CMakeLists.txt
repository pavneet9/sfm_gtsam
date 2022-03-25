add_executable(main main.cpp)
target_link_libraries(main sfm_pipeline visualizer ${opencv_libs} ${PCL_LIBRARIES} gtsam)