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

/* hold grid dimensions and indexes for both serial and
 * parallel cases */
struct dimensions_t {
    int total;
    int local;
    int rest;
    int nx_rest;
    int nx_local;
    std::pair<int, int> local_i;
    int start_i;
    int end_i;
};


class Solver {
  
private:

    /* number of parallel processes */
    int num_tasks;

    /* discrete grid dimensions */
    int nx,ny,nz;
    dimensions_t dims;

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

    /* preparar vectores *after y *before */
    void prepare_solution(const int, const int,
            boost_array3d_t& solution, boost_array2d_t& before, boost_array2d_t& after);
  
public:
  
  /* variables to keep track of timing */
  double init_time;
  double linear_time; 
  double nonlinear_time;
  
  Solver(const int nx, const int ny, const int nz, 
    const double dX, const double dT);
  
  ~Solver();

  static dimensions_t get_local_dimensions(const int nx, const int ny, const int nz);
  
  /* solver initialization */
  void start(const int task_id, BrainModel& brain, const int method);
  
  
  /* solution methods */
  void lie_trotter(const double current_time, 
                const int task_id, 
                boost_array3d_t& solution,
                BrainModel& brain_model);

  void expl(const int task_id, const int num_tasks,
            boost_array3d_t& solution, BrainModel& brain);

};

# endif