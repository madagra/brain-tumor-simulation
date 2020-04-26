# include <iostream> 
# include <cstdlib>
# include <cmath>      
# include <vector>     
# include <fstream>   
# include <sstream>   
# include <iomanip>
# include <string>
# include <vector>  

# if defined _MPI
# include <mpi.h>
# endif

# include "matrix.h"

using namespace std;

# if not defined BRAIN_H
# define BRAIN_H

class BrainModel {
  
private:

    /* diffusion coefficients for white and grey matter */
    double D_white = 0.255;
    double D_grey = 0.051;

    /* min/max values of the diffusion to distinguish between 
    * white and grey matter */
    static constexpr auto is_grey = std::make_pair(75, 110);
    static constexpr auto is_white = std::make_pair(110, 225);   

    /* tumor size and initial position */
    int tumor_size;
    int tumor_position[3];

      /* input file name */
    std::string fname;

public:
    
    /* Maximum cellular density 
     * [C] = cells/mm^3 */
    double Cm; 

    /* Net cellular proliferation ratio for white or grey matter
     * [rho] = cells/day */
    double rho;

    /* radiotherapy intensity */
    double dose;
    double t_radio;

    /* radiotherapy coefficients */
    double alpha;
    double beta;
    double s;

    /* Constructors */
    BrainModel();

    BrainModel(const double, 
               const double, 
               const double, 
               const double, 
               const double, 
               const double, 
               const double, 
               const int, 
               const double);

    /* methods */
    void read(matrix3D *, 
              int&, 
              int&, 
              int&, 
              const int);

    void init_tumor(const int, 
                    matrix3D *, 
                    const int, 
                    const int);

    double norm_tumor(const int, 
                      const int, 
                      matrix3D *);

};

# endif