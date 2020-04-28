# include <sstream>   
# include <iomanip>
# include <string>
# include <chrono>
# include <sys/types.h>
# include <sys/stat.h>
# include <sys/time.h>
# include <omp.h>

# include "brain.hpp"
# include "matrix.hpp"
# include "solver.hpp"

using namespace std;

# define TEST_DIM 32

/* variables para el MPI/OpenMP */
int task_id;
int num_tasks;
int num_threads;
int provided;
int required;

/* arrays to hold the solution of the equation */
boost_array3d_t solution;
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
double dX = 1.0;
double dT = 0.0001;
double total_time = 30.0;
int dump = 20;
int method = 0;

int main( int argc, char *argv[] ) {
  
    double t_radio_ = 0.0;
    ostringstream fname;

    /* medir los tiempos */
    struct timeval tp;
    int t_elap;
    double t1,t2;

    task_id   = 0;
    num_tasks = 1;

    num_threads = omp_get_max_threads();  
    
    cout << " # of threads: " << num_threads << endl;
    cout << " Input spacial and temporal steps: " << endl;
    cin >> dX >> dT;
    cout << " Input total simulation time: " << endl;
    cin >> total_time;
    cout << " Input every how many steps to input the solution: " << endl;
    cin >> dump;
    cout << " Input the time step when to start radiotherapy " << endl;
    cin >> t_radio_;
    cout << " Input the solving method (0: Lie-Trotter, 1: explicit) " << endl;
    cin >> method;
    
    /* variables y metodos relacionados con el cerebro */
    BrainModel brain_model; 
    
    /* radiotherapy starting time step */
    brain_model.t_radio = t_radio_;
    
    /* initialize the brain model with default parameters
    * and read the diffusion coefficients matrix */

    brain_model.read();
    nx = (*brain_model.diffusion).shape()[0];
    ny = (*brain_model.diffusion).shape()[1];
    nz = (*brain_model.diffusion).shape()[2];

    double norma = l2_norm((*brain_model.diffusion));

    cout << " Spacial dimension of the grid defining the brain: " << nx << " " << ny << " " 
    << nz << " " << dim_tot << " " << dim_rest << endl;
    cout << " Write solution every " << dump << " steps" << endl;
    if(method == 0) 
        cout << " Starting Lie-Trotter solver " << endl;
    else if(method == 1) 
        cout << " Starting explicit solver " << endl;

    // fname.str("");
    // fname.clear();
    // fname << "./slice_brain.vtk";
    // print_vtk(brain, fname.str().c_str(), 30);
    
    /* definicion de las dimensiones por cada proceso */
    dim_tot   = nx*ny*nz;
    dim_local = (nx/num_tasks)*ny*nz;
    dim_rest  = (nx%num_tasks)*ny*nz;
    nx_rest   = nx%num_tasks;
    nx_local  = nx/num_tasks;
  
    /* initialize tumor area */  
    is = 0;
    ie = nx;
    // C.reset_dimension(nx_local,ny,nz);


    // TODO use a smart pointer for the solution array as below
    // std::unique_ptr<boost_array3d_t> solution(new boost_array2d_t(boost::extents[nx][ny][nz]));
    boost_array3d_t solution(boost::extents[nx][ny][nz]);
    solution = brain_model.init_tumor(nx, ny, nz);

    std::cout << std::endl;
    std::cout << " Total brain diffusion coefficient " << l2_norm(*brain_model.diffusion) << " " << std::endl;
    std::cout << " Initial tumor concentration " << l2_norm(solution) << " " << std::endl;

    Solver solver(nx, ny, nz, dX, dT);
    solver.start(task_id, brain_model, method);

    /* hold the partial solution for later visualizion of the tumor spread */
    std::vector<double> Ct_ = std::vector<double>(dim_tot, 0.0);
    int status = mkdir("./solution", S_IRWXU | S_IRWXG);	  

    Ct = new double[dim_tot];
    for(int i = 0; i < dim_tot; i++)
        Ct[i] = 0.0;
    
    int i_dump = 0;
    double norm   = 0.0;
    double t_norm = 0.0;
    bool radio_active = false;
    
    double simulation_time = 0.0;

    
    int t  = 0;
    while(t < total_time/dT) {
    
        double current = t*dT;

        if(current >= brain_model.t_radio && radio_active == false) {
            std::cout << " Turning on radiotherapy effect at time = " << current << std::endl;
            radio_active = true;
        }
        
        t1 = 0.0; t2 = 0.0;
        t_elap = gettimeofday(&tp, NULL);
        t1 = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
        
        // solve the system of equation at current time
        if(method == 0) {
            solver.lie_trotter(current, task_id, *brain_model.diffusion, solution, brain_model);
        }
        else if(method == 1) {
            // solver.expl(task_id,num_tasks,&D,&C, &brain_model);
        }

        t_elap = gettimeofday(&tp, NULL);
        t2     = (double)tp.tv_sec+(1.e-6)*tp.tv_usec;    
        simulation_time += (t2-t1);  
   
        std::cout << std::setprecision(6) << fixed << " Tumor concentration = "
            << " at time " << current + dT << " = " << l2_norm(solution) 
            << " (" << sum_elements(solution) << ")" << std::endl;
               
        if( t % dump == 0 && t > 0) {  
            // C.fill(Ct);
            fname.str("");
            fname.clear();
            fname << "./solution/concentration_plane" << i_dump << ".vtk";
            std::cout << " Printing intermediate solution at: " << fname.str().c_str() << std::endl;
            // print_vtk(Ct,fname.str().c_str(),30);
            ++i_dump;
        }

        // go to the next time step
        ++t;
    }

    return 0;
  
    int simulation_time_;
    simulation_time_ = simulation_time;

    cout << " \n TIMING \n ";
    cout << setprecision(6) << fixed << " Tiempo initializacion = " << solver.init_time << endl;
    cout << setprecision(6) << fixed << "  Tiempo solucion del sistema = " << simulation_time << endl;
    if(method == 0) {
        cout << setprecision(6) << fixed << " Tiempo medio parte linear = " << solver.linear_time << endl;
        cout << setprecision(6) << fixed << " Tiempo medio parte non linear = " << solver.nonlinear_time << endl;
    }
   
    if(task_id == 0)
        delete[] Ct;
    
    return 0;
  
}
