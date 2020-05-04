#include <iostream> 
#include <cstdlib>
#include <cmath>
#include <fstream>   
#include <sstream>   
#include <iomanip>
#include <string>
#include <boost/multi_array.hpp>

# if not defined MATRIX_CLASS_H
# define MATRIX_CLASS_H

using namespace std;

/* predefined Boost multi-array types */
typedef boost::multi_array<double, 3> boost_array3d_t;
typedef boost::array<boost_array3d_t::index, 3> boost_array3d_ind_t;
typedef boost::multi_array<double, 2> boost_array2d_t;
typedef boost::multi_array<double, 1> boost_array1d_t;

/* print 3D array in Paraview format */
void print_vtk(boost_array3d_t&, string, const int);
void print_vtk(std::vector<double>&, const int, const int, const int, string);

/* print 3D array in standard csv format */
void print_array_slice(boost_array3d_t &arr, string filename, int slice_ind);
void print_array_full(boost_array3d_t &arr, string filename);

double l2_norm(boost_array3d_t& data);
double sum_elements(boost_array3d_t& data);

# endif