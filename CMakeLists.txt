# general configuration
cmake_minimum_required (VERSION 3.13)
project(brain_tumor_simulation VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include Boost library
find_package(Boost)
if(NOT Boost_FOUND)
  message(FATAL_ERROR "Cannot find the Boost library!") 
endif()
include_directories(${Boost_INCLUDE_DIRS})

# compilation
set(SOURCES
    src/brain.cpp
    src/solver.cpp
    src/main.cpp
)
add_executable(run_simulation ${SOURCES})
target_include_directories(run_simulation
	PRIVATE
	${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(run_simulation 
	PRIVATE
	${Boost_LIBRARIES}
)
