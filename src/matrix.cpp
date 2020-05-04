# include <sstream>   

# include "matrix.hpp"

using namespace std;

double l2_norm(boost_array3d_t& arr) {
    double norm = 0.0;
    const auto sup(arr.data() + arr.num_elements());
    for (auto it = arr.data(); it != sup; ++it) {
        norm += pow(*it, 2);
    }
    return sqrt(norm);
};

double sum_elements(boost_array3d_t& arr) {
    double sum = 0.0;
    const auto sup(arr.data() + arr.num_elements());
    for (auto it = arr.data(); it != sup; ++it) {
        sum += *it;
    }
    return sum;
}

void print_array_slice(boost_array3d_t &arr, string filename, int slice_ind) {
    ofstream f;
    f.open(filename.c_str(), ofstream::out);
    f << "x,y,value\n";
    for( int j = 0; j < arr.shape()[1]; j++) {
        for( int i = 0; i < arr.shape()[0]; i++) {
            f << std::setprecision(6) << fixed << i << "," << j << "," << arr[i][j][slice_ind] << std::endl;
        }
    }    
    f.close();
}

void print_array_full(boost_array3d_t &arr, string filename) {
    ofstream f;
    f.open(filename.c_str(), ofstream::out);
    f << "x,y,z,value\n";
    for( int i = 0; i < arr.shape()[0]; i++) {
        for( int j = 0; j < arr.shape()[1]; j++) {
            for(int k = 0; k < arr.shape()[2]; k++) {
                f << std::setprecision(6) << fixed << i << "," << j << "," << k << "," << arr[i][j][k] << std::endl;
            }
        }
    }
    f.close();
}

/* print in Paraview accepted format */
void print_vtk(boost_array3d_t &C, string filename, const int z_ind)
{

    ofstream f;
    int nx = C.shape()[0];
    int ny = C.shape()[1];

    f.open(filename.c_str(),ofstream::out);
    f << "# vtk DataFile Version 2.0\n";
    f << "Comment goes here\n";
    f << "ASCII\n";
    f << std::endl;
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << nx << " " << ny << " " << 1 << std::endl;
    f << std::endl;
    f << "ORIGIN  "  << 0.0 << " " << 0.0 << " " << 0.0 << std::endl;
    f << "SPACING "  << 1.0  << " " << 1.0  << " " << 0.0 << std::endl;
    f << std::endl;
    f << "POINT_DATA " << nx*ny << std::endl;
    f << "SCALARS scalars float\n" ;
    f << "LOOKUP_TABLE default\n";
    f << std::endl;  
    for( int j = 0; j < ny; j++) {
        for( int i = 0; i < nx; i++) {
            f << C[i][j][z_ind] << std::endl;
        }
    }
    f.close();
}
