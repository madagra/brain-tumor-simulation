# include <cstdlib>
# include <cmath>
# include <sys/time.h>
#include <boost/multi_array.hpp>

# include "matrix.hpp"
# include "brain.hpp"

# define CONV_UP  1.0e-6  /* parametro de convergencia Jacobi */
# define CONV_DO  1.0e2   /* parametro para chequear si Jacobi explota */
# define ITER_MAX 100

# define BTAG 0 
# define ATAG 1

# if not defined SOLVER_H
# define SOLVER_H

class Solver {
  
private:

    /* number of parallel processes */
    int num_tasks;

    /* discrete grid dimensions */
    int nx,ny,nz;
    int nx_loc;
    int dim_tot;
    int dim_rest;
    int nx_rest;

    /* Jacobi matrix coefficients */
    boost_array2d_t *coeffs;
    int dim_coeffs;
    int s_order;

    /* space and time step values */
    double dX;
    double dT;

    /* timing */
    struct timeval tp;
    int t_elap;
    double t1, t2;

    /* initialize dimensions in the serial and parallel cases */
    void initialize_dimensions(const int nx, const int ny, const int nz);

    /* get starting/ending indices on the full grid depending on the current rank */
    std::pair<int, int> get_local_indices(const int task_id);

    /* preparar vectores *after y *before */
    void prepare_solution(const int, const int , const int, const int, const int,
            matrix3D *, matrix2D *, matrix2D *);
  
public:
  
  /* variables to keep track of timing */
  double init_time;
  double linear_time; 
  double nonlinear_time;
  
  Solver(const int nx, const int ny, const int nz, 
    const double dX, const double dT);
  
  ~Solver();
  
  /* initializacion parte lineal Lie-Trotter o explicito */
  void start(const int task_id, BrainModel& brain, const int method);
  
  /* solucion del sistema */
  void lie_trotter(const double current_time, 
                const int task_id, 
                boost_array3d_t&, 
                boost_array3d_t&,
                BrainModel&);

  void expl(const int, const int, matrix3D *, matrix3D *, BrainModel *);

};

# endif