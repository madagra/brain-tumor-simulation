/* librerias standard */
# include <cstdlib>
# include <cmath>
# include <sys/time.h>

/* librerias locales */
# include "matrix.hpp"
# include "brain.hpp"

# define CONV_UP  1.0e-6  /* parametro de convergencia Jacobi */
# define CONV_DO  1.0e2   /* parametro para chequear si Jacobi explota */
# define ITER_MAX 100

# define BTAG 0 
# define ATAG 1

# if not defined JACOBI_H
# define JACOBI_H

class classSystem {
  
private:
  
  /* dimensiones de la grilla */
  int nx,ny,nz;
  int nx_loc;
  int dim_tot;
  int dim_rest;
  int nx_rest;
  
  /* dimensiones matriz coeficientes Jacobi */
  double **coeffs;
  int dim_coeffs;
  int s_order;
  
  /* pasos en tiempo y espacio */
  double dX;
  double dT;
  
  /* posiciones local de la grilla */
  int is;
  int ie;

  /* medir los tiempos */
  struct timeval tp;
  int t_elap;
  double t1, t2;
  
  /* preparar vectores *after y *before */
  void prepare_solution(const int, const int , const int, const int, const int,
			matrix3D *, matrix2D *, matrix2D *);
  
public:
  
  /* tiempos */
  double init_time;
  double linear_time; 
  double nonlinear_time;
  
  /* constructores y destructores */
  classSystem();
  
  classSystem(const int, const int, const int, 
	      const int, const int, const double, const double);
  
  ~classSystem();
  
  /* initializacion parte lineal Lie-Trotter o explicito */
  void start(matrix3D *, BrainModel *, const int);
  
  /* solucion del sistema */
  void lie_trotter(const double, const int, const int, matrix3D *, matrix3D *, BrainModel *); 
  void expl(const int, const int, matrix3D *, matrix3D *, BrainModel *);

};

# endif