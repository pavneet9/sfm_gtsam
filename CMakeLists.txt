cmake_minimum_required(VERSION 3.1) # Mandatory.
project( Surf_Detector ) # Mandatory.

# Set build type if not set.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib)

# Create a results dir in project root where we will put the results of the
# build
set(RESULTS_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/bin)



find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
  message(STATUS "OpenCV directories: ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "OpenCV not found, please read the README.md")
endif(OpenCV_FOUND)

find_package(PCL REQUIRED)
if(PCL_FOUND)
  message(STATUS "Found PCL")
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
else()
  message(FATAL_ERROR "PCL not found, please read the README.md")
endif(PCL_FOUND)


find_package(Ceres REQUIRED)

find_package(GTSAM REQUIRED) # Uses installed package
include_directories(${GTSAM_INCLUDE_DIR})
link_directories(
    ${GTSAM_LIBRARY_DIRS}
)
# After all are setup is done, we can go to our src/ directory to build our
# files
add_subdirectory("src")


