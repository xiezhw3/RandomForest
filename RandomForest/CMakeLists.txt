cmake_minimum_required(VERSION 3.0)
project(ITK_MPI)

find_package(MPI REQUIRED)
add_definitions(-std=c++11)

include_directories(${MPI_INCLUDE_PATH})

file(GLOB_RECURSE main_SOURCES "RandomForest/*.cpp")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable (main ${main_SOURCES})
target_link_libraries(main ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(main PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(main PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()