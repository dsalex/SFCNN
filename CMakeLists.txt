project(SFCNN)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Boost COMPONENTS python REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
find_package(Armadillo REQUIRED)

include_directories("." ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
#Need FindNumPy
include_directories("/usr/lib64/python2.7/site-packages/numpy/core/include")
include_directories("/usr/include/python2.7/")
include_directories("/usr/local/include")


file(GLOB_RECURSE INC_ALL "*.h*")
file(GLOB_RECURSE SRC_ALL "*.c*")

aux_source_directory(. SRC_LIST)

#add_executable(${PROJECT_NAME} ${SRC_LIST} ${INC_ALL} ${SRC_ALL})
add_library(${PROJECT_NAME} SHARED ${SRC_LIST} ${INC_ALL} ${SRC_ALL})

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} "/usr/local/lib/libboost_numpy.so")

