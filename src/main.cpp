# include <sstream>   
# include <iomanip>
# include <string>
# include <chrono>
# include <sys/types.h>
# include <sys/stat.h>
# include <sys/time.h>
#include <errno.h>
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

/* array to hold the solution of the equation */
boost_array3d_t solution;
std::string solution_folder = "./solution";

/* grid dimensions */
int nx, ny, nz;

/* input parameters */
double dX, dT;
double total_time;
int dump;
int method;

int main( int argc, char *argv[] ) {
  
    double t_radio_ = 0.0;
    std::ostringstream fname;

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

    /* folder to hold the partial solutions */
    int status_mkdir = mkdir(solution_folder.c_str(), S_IRWXU | S_IRWXG);
    if (0 != status_mkdir && errno != EEXIST) {
        std::cout << " Cannot create directory to store partial solutions" << std::endl;
        status_mkdir = -1;
    } else {
        status_mkdir = 0;
    }
    
    /* initialize the brain model with default parameters
    * and read the diffusion coefficients matrix */

    brain_model.read();

    print_array_slice((*brain_model.diffusion), solution_folder + "/diffusion_slice.csv", 30);
    nx = (*brain_model.diffusion).shape()[0];
    ny = (*brain_model.diffusion).shape()[1];
    nz = (*brain_model.diffusion).shape()[2];

    std::cout << " Dimensions of the spacial grid: " << nx << " " << ny << " " << nz << std::endl;
    std::cout << " Write solution every " << dump << " steps" << std::endl;
    if(method == 0) 
        std::cout << " Starting Lie-Trotter solver " << std::endl;
    else if(method == 1) 
        std::cout << " Starting explicit solver " << std::endl;

    boost_array3d_t solution(boost::extents[nx][ny][nz]);
    solution = brain_model.init_tumor(nx, ny, nz);

    std::cout << std::endl;
    std::cout << " Total brain diffusion coefficient " << l2_norm(*brain_model.diffusion) << " " << std::endl;
    std::cout << " Initial tumor concentration " << l2_norm(solution) << " " << std::endl;

    Solver solver(nx, ny, nz, dX, dT);
    solver.start(task_id, brain_model, method);

    int i_dump = 0;
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
            solver.lie_trotter(current, task_id, solution, brain_model);
        }
        else if(method == 1) {
            solver.expl(task_id,num_tasks, solution, brain_model);
        }

        t_elap = gettimeofday(&tp, NULL);
        t2     = (double)tp.tv_sec+(1.e-6)*tp.tv_usec;    
        simulation_time += (t2-t1);  
   
        std::cout << std::setprecision(6) << fixed << " Tumor concentration = "
            << " at time " << current + dT << " = " << l2_norm(solution) 
            << " (" << sum_elements(solution) << ")" << std::endl;
               
        if( t % dump == 0 && status_mkdir == 0) {  
            fname.str("");
            fname.clear();
            fname << solution_folder <<"/slice_" << i_dump << ".csv";
            std::cout << " Printing intermediate solution at: " << fname.str().c_str() << std::endl;
            print_array_slice(solution, fname.str().c_str(), 30);
            ++i_dump;
        }

        // advance to the next time step
        ++t;
    }
 
    int simulation_time_;
    simulation_time_ = simulation_time;

    cout << " \n TIMING \n ";
    cout << setprecision(6) << fixed << " Initialization time = " << solver.init_time << " s" << endl;
    cout << setprecision(6) << fixed << "  Solver time = " << simulation_time  << " s" << endl;
    if(method == 0) {
        cout << setprecision(6) << fixed << " Average time Lie-Trotter linear part = " << solver.linear_time << " s" << endl;
        cout << setprecision(6) << fixed << " Average time Lie-Trotter non-linear part = " << solver.nonlinear_time << " s" << endl;
    }
    
    return EXIT_SUCCESS;
}
