# include <sstream>   
# include <iomanip>
# include <string>

/* librearias para medir tiempos y crear carpetas */
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

/* libreria para el MPI y OpenMP */
# if defined _MPI
# include <mpi.h>
# endif

# if defined _OMP
# include <omp.h>
# endif

/* librerias locales */
# include "brain.hpp"
# include "matrix.hpp"
# include "solver.hpp"


using namespace std;

# define TEST_DIM 32

void print_vtk(double *, string);
void print_vtk(double *, string, const int);
void print_vtk(matrix3D *, string, const int);

/* variables para el MPI/OpenMP */
int task_id;
int num_tasks;
int num_threads;
int provided;
int required;

/* solucion de la ecuacion */
matrix3D C;
double *Ct, *p_Ct;

/* coefficiente de difusion */
matrix3D D;

/* dimensiones del problema */
int dim_tot;
int dim_local;
int dim_rest;
int nx, ny, nz;
int nx_local,nx_rest;

/* direcciones de la grilla por cada procesadora */
int is;
int ie;

/* parametros de metodo iterativo */
double dX       = 1.0;
double dT       = 0.0001;
double TOT_TIME = 30.0;
int dump        = 20;
int method      = 0;

int main( int argc, char *argv[] ) {
  
  double t_radio_ = 0.0;
  ostringstream fname;
  
  /* medir los tiempos */
  struct timeval tp;
  int t_elap;
  double t1,t2;
  
# if defined _MPI
  required = MPI_THREAD_SINGLE; /* solo el master thread puede llamar MPI */
  MPI_Init_thread(&argc,&argv,required,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&task_id);
  MPI_Comm_size(MPI_COMM_WORLD,&num_tasks);
# if defined _OMP
  if(provided < required) {  /* chequear el nivel de threading */
    omp_set_num_threads(1);
  }
# endif
# else
  task_id   = 0;
  num_tasks = 1;
# endif  
  
# if defined _OMP
  num_threads = omp_get_max_threads();  
# else
  num_threads = 1;	
# endif
  
# if defined TEST
  int i,j,k;
  ofstream f;
  f.open("Cerebro_test.csv",ofstream::out);
  for(i = 1; i <= TEST_DIM; i++) {
    for(j = 1; j <= TEST_DIM; j++) {
      for(k = 1; k <= TEST_DIM; k++) {
	double val = i*j*k;
	f << i << "," << j << "," << k << "," << val << endl;
      }
    }
  }
  f.close();
# endif
    
  if(task_id == 0) {
    
    cout << " # de procesos MPI: " << num_tasks << endl;
    cout << " # de threads: " << num_threads << endl;
    cout << " Ingresar pasos espacial y temporal: " << endl;
    cin >> dX >> dT;
    cout << " Ingresar el tiempo total: " << endl;
    cin >> TOT_TIME;
    cout << " Ingresar cada cuantos pasos de tiempo imprimir la solucion: " << endl;
    cin >> dump;
    cout << " Ingresar el tiempo de inicio de la radioterapia: " << endl;
    cin >> t_radio_;
    cout << " Ingresar el method: " << endl;
    cin >> method;
  }
 
# if defined _MPI
  MPI_Bcast( &dX,1,MPI_DOUBLE,0,MPI_COMM_WORLD );
  MPI_Bcast( &dT,1,MPI_DOUBLE,0,MPI_COMM_WORLD );
  MPI_Bcast( &TOT_TIME,1,MPI_DOUBLE,0,MPI_COMM_WORLD );
  MPI_Bcast( &dump,1,MPI_INT,0,MPI_COMM_WORLD );
  MPI_Bcast( &t_radio_,1,MPI_DOUBLE,0,MPI_COMM_WORLD );
  MPI_Bcast( &method,1,MPI_INT,0,MPI_COMM_WORLD );
# endif   
 
  /* variables y metodos relacionados con el cerebro */
  BrainModel brain; 
  
  /* radiotherapy starting time step */
  brain.t_radio = t_radio_;
  
  /* initialize the brain model with default parameters
   * and read the diffusion coefficients matrix */
  brain.read(&D, nx, ny, nz, task_id);
  cout << " nx: " << nx << endl;
  cout << " ny: " << ny << endl;
  cout << " nz: " << nz << endl;
  return 0;

  fname.str("");
  fname.clear();
  fname << "./slice_brain.vtk";
  print_vtk(&D,fname.str().c_str(),30);
  
  /* definicion de las dimensiones por cada proceso */
  dim_tot   = nx*ny*nz;
  dim_local = (nx/num_tasks)*ny*nz;
  dim_rest  = (nx%num_tasks)*ny*nz;
  nx_rest   = nx%num_tasks;
  nx_local  = nx/num_tasks;
  
  /* definicion de los indices locales de la grilla */
# if defined _MPI
  if(task_id < num_tasks - 1) {
    is =  task_id*nx_local;
    ie = (task_id+1)*nx_local;
  }
  else {
    is = task_id*nx_local;
    ie = nx;
    nx_local += nx_rest;
  }
# else
  is = 0;
  ie = nx;
# endif
  
  /* creacion del tumor */  
  C.reset_dimension(nx_local,ny,nz);
  
  /* solucion total para imprimir */
  if(task_id == 0) {
     Ct = new double[dim_tot];
     for(int i = 0; i < dim_tot; i++)
       Ct[i] = 0.0;
  }
# if defined _MPI
  p_Ct = new double[nx_local*ny*nz];
  for(int i = 0; i < nx_local*ny*nz; i++)
    p_Ct[i] = 0.0;
# endif
  
  brain.init_tumor(task_id,&C,is,ie);
    
  if(task_id == 0) {  
    cout << " Dimensiones de la grilla espacial: " << nx << " " << ny << " " 
    << nz << " " << dim_tot << " " << dim_rest << endl;
    cout << " Dimensiones locales (nx/nx_rest): " << nx_local << " " << nx_rest << endl;
    cout << " Escribiendo la solucion cada " << dump << " pasos." << endl;
    cout << " Lectura completada! " << endl;
    if(method == 0) 
      cout << " Empezando Lie-Trotter " << endl;
    if(method == 1) 
      cout << " Empezando explicito " << endl;
  }
  
  classSystem solve(nx,ny,nz,num_tasks,task_id,dX,dT);
  solve.start(&D, &brain, method);
  
  if(task_id == 0) cout << " initializacion sistema completada! " << endl;
  
# if defined _MPI
  MPI_Barrier(MPI_COMM_WORLD);
# endif  
  
  /* solucion de la ecuacion con Jacobi */
  int status;
  if(task_id == 0) 
    status = mkdir("./solution", S_IRWXU | S_IRWXG);	  
  
  int t  = 0;
  int i_dump = 0;
  double norm   = 0.0;
  double t_norm = 0.0;
  bool radio_active = false;
  
  double simulation_time = 0.0;
    
  while(t*dT < TOT_TIME) {
    
    double TIME = t*dT;
    if(TIME >= brain.t_radio && radio_active == false && task_id == 0) {
       cout << " Radioterapia activada " << endl;
       radio_active = true;
    }
    
    t1 = 0.0; t2 = 0.0;
    t_elap = gettimeofday(&tp, NULL);
    t1     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
    
    if(method == 0) {
       solve.lie_trotter(TIME,task_id,num_tasks,&D,&C,&brain);
    }
    else if(method == 1) {
       solve.expl(task_id,num_tasks,&D,&C,&brain);
    }
    
    t_elap = gettimeofday(&tp, NULL);
    t2     = (double)tp.tv_sec+(1.e-6)*tp.tv_usec;    
    simulation_time += (t2-t1);  

    norm = 0.0;
    t_norm = C.sum_elements();
# if defined _MPI
    MPI_Allreduce(&t_norm,&norm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
# else
    norm = t_norm;
# endif
    if(task_id == 0)
      cout << setprecision(6) << fixed << " Concentracion tumor = " << norm << " al tiempo = " << t+1 << endl;
    
    if( t%dump == 0 && t > 0) {  

      norm = 0.0;
# if defined _MPI
      int ierr;
      C.fill(p_Ct);
      t_norm = C.sum_elements();
    
      ierr = MPI_Gather(p_Ct, dim_local, MPI_DOUBLE, 
	                Ct, dim_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      ierr = MPI_Allreduce(&t_norm,&norm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      
      if( ierr != MPI_SUCCESS) {
         cout << " ERROR parallel version. The program will stop! " << endl;
         MPI_Finalize();
         exit(EXIT_FAILURE);
      }
      
# else
      C.fill(Ct);
      norm = t_norm;
# endif

      if(task_id == 0) {
        	
	fname.str("");
	fname.clear();
	fname << "./solution/concentration_plane" << i_dump << ".vtk";
	cout << " Imprimiendo solucion en: " << fname.str().c_str() << endl;
	print_vtk(Ct,fname.str().c_str(),30);
	
      }
      ++i_dump;
    }
    
    ++t;
  }
  
   int simulation_time_;
   simulation_time_ = simulation_time;
# if defined PARALLEL
   MPI_Reduce(&simulation_time_,&simulation_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
# endif

   if(task_id == 0) {
    cout << " \n TIMING \n ";
    cout << setprecision(6) << fixed << " Tiempo initializacion = " << solve.init_time << endl;
    cout << setprecision(6) << fixed << "  Tiempo solucion del sistema = " << simulation_time << endl;
    if(method == 0) {
       cout << setprecision(6) << fixed << " Tiempo medio parte linear = " << solve.linear_time << endl;
       cout << setprecision(6) << fixed << " Tiempo medio parte non linear = " << solve.nonlinear_time << endl;
    }
   }
   
# if defined _MPI
  delete[] p_Ct;
# endif
  if(task_id == 0)
     delete[] Ct;
  
# if defined _MPI
  MPI_Finalize(); 
# endif
  return 0;
  
}

void print_vtk(double *C, string filename)
  {

    ofstream f;

    f.open(filename.c_str(),ofstream::out);
    f << "# vtk DataFile Version 2.0\n";
    f << "Comment goes here\n";
    f << "ASCII\n";
    f << endl;
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << nx << " " << ny << " " << nz << endl;
    f << endl;
    f << "ORIGIN  "  << 0.0 << " " << 0.0 << " " << 0.0 << endl;
    f << "SPACING "  << 1.0  << " " << 1.0  << " " << 1.0 << endl;
    f << endl;
    f << "POINT_DATA " << nx*ny*nz << endl;
    f << "SCALARS scalars float\n" ;
    f << "LOOKUP_TABLE default\n";
    f << endl;  
    for( int k = 0; k < nz; k++) {
       for( int j = 0; j < ny; j++) {
          for( int i = 0; i < nx; i++) {
            f << C[k+nz*j+ny*nz*i] << endl;
          }
       }
    }
    f.close();
  } 
  
void print_vtk(double *C, string filename, const int z_ind)
  {

    ofstream f;

    f.open(filename.c_str(),ofstream::out);
    f << "# vtk DataFile Version 2.0\n";
    f << "Comment goes here\n";
    f << "ASCII\n";
    f << endl;
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << nx << " " << ny << " " << 1 << endl;
    f << endl;
    f << "ORIGIN  "  << 0.0 << " " << 0.0 << " " << 0.0 << endl;
    f << "SPACING "  << 1.0  << " " << 1.0  << " " << 0.0 << endl;
    f << endl;
    f << "POINT_DATA " << nx*ny << endl;
    f << "SCALARS scalars float\n" ;
    f << "LOOKUP_TABLE default\n";
    f << endl;  
    for( int j = 0; j < ny; j++) {
       for( int i = 0; i < nx; i++) {
         f << C[z_ind+nz*j+ny*nz*i] << endl;
       }
    }
    f.close();
  } 
  
void print_vtk(matrix3D *C, string filename, const int z_ind)
  {

    ofstream f;

    f.open(filename.c_str(),ofstream::out);
    f << "# vtk DataFile Version 2.0\n";
    f << "Comment goes here\n";
    f << "ASCII\n";
    f << endl;
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << nx << " " << ny << " " << 1 << endl;
    f << endl;
    f << "ORIGIN  "  << 0.0 << " " << 0.0 << " " << 0.0 << endl;
    f << "SPACING "  << 1.0  << " " << 1.0  << " " << 0.0 << endl;
    f << endl;
    f << "POINT_DATA " << nx*ny << endl;
    f << "SCALARS scalars float\n" ;
    f << "LOOKUP_TABLE default\n";
    f << endl;  
    for( int j = 0; j < ny; j++) {
       for( int i = 0; i < nx; i++) {
         f << C->get(i,j,z_ind) << endl;
       }
    }
    f.close();
  } 