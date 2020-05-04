# include "solver.hpp"


Solver::Solver(const int nx, const int ny, const int nz, 
            const double dX, const double dT) {
 
    /* time and space steps */
    this->dT = dT;
    this->dX = dX;
    
    /* full grid dimensions */
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;

    /* local grid dimensions and indices */
    this->dims = Solver::get_local_dimensions(nx, ny, nz);

    /* set Jacobi order and initialize coefficients */
    s_order = 7;
    dim_coeffs = this->dims.nx_local * ny * nz;
    coeffs = new boost_array2d_t(boost::extents[dim_coeffs][s_order]);
    std::fill_n((*coeffs).data(), (*coeffs).num_elements(), 0.0);

    /* keep track of elapsed times */
    this->init_time = 0.0;
    this->linear_time = 0.0;
    this->nonlinear_time = 0.0;
  
}

Solver::~Solver() {
  delete coeffs;
}

dimensions_t Solver::get_local_dimensions(int nx, int ny, int nz) {

    int num_tasks;
    int task_id;
    dimensions_t dims;

# if defined _MPI
    MPI_Comm_size(MPI_COMM_WORLD,&num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&task_id);
# else
    num_tasks = 1;
    task_id = 0;
# endif

    /* local dimensions and indices */
    dims.total = nx * ny * nz;
    dims.local = (nx / num_tasks) * ny * nz;
    dims.rest  = (nx % num_tasks) * ny * nz;
    dims.nx_rest   = nx % num_tasks;
    dims.nx_local  = nx / num_tasks;   

    int start_i, end_i;
    if(task_id < num_tasks - 1) {
        start_i =  task_id * dims.nx_local;
        end_i = (task_id+1) * dims.nx_local;
    }
    else {
        start_i = task_id * dims.nx_local;
        end_i = nx;
    }
    dims.local_i = std::make_pair(start_i, end_i);

    return dims;
}

void Solver::start(const int task_id, BrainModel& brain, const int method) {
  
    double D_loc;
    int ind = 0;
    double t_coeff = dT / (2.0 * pow(dX, 2) );
    double rho_;

    int is, ie;
    std::tie(is, ie) = this->dims.local_i;

    t_elap = gettimeofday(&tp, NULL);
    t1     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
          
    for(int i = is; i < ie; i++) {
        for(int j = 1; j < ny-1; j++) {
            for(int k = 1; k < nz-1; k++) {    
        
                D_loc = (*brain.diffusion)[i][j][k];
                ind = (i-is)*ny*nz + j*nz + k;

                if( D_loc > 0.0 ) {
                    rho_ = brain.rho + brain.s - 1.0;
                } 
                else {
                    rho_ = 0.0;  
                }
	  
                if( i > 0 && i < nx - 1) {

                    (*coeffs)[ind][1] = t_coeff * ( (*brain.diffusion)[i+1][j][k] + D_loc );
                    (*coeffs)[ind][2] = t_coeff * ( (*brain.diffusion)[i-1][j][k] + D_loc );
                    (*coeffs)[ind][3] = t_coeff * ( (*brain.diffusion)[i][j+1][k] + D_loc );
                    (*coeffs)[ind][4] = t_coeff * ( (*brain.diffusion)[i][j-1][k] + D_loc );
                    (*coeffs)[ind][5] = t_coeff * ( (*brain.diffusion)[i][j][k+1] + D_loc );
                    (*coeffs)[ind][6] = t_coeff * ( (*brain.diffusion)[i][j][k-1] + D_loc );

                    if(method == 0) {
                        (*coeffs)[ind][0] = 1.0; /* 1.0 + */
                        for(int jj = 1; jj < 7; jj++) {
                            /* all other coefficients */
                            (*coeffs)[ind][0] = (*coeffs)[ind][0] + (*coeffs)[ind][jj]; 
                        }
                    }
                    else if(method == 1) {
                        (*coeffs)[ind][0] = 1.0; /* 1.0 + */
                        for(int jj = 1; jj < 7; jj++) {
                            /* all other coefficients */
                            (*coeffs)[ind][0] = (*coeffs)[ind][0] - (*coeffs)[ind][jj];
                        }
                        (*coeffs)[ind][7] = dT * rho_;
                    }
                }	  
            }
        }
    }
    
    t_elap = gettimeofday(&tp, NULL);
    t2     = (double)tp.tv_sec+(1.e-6)*tp.tv_usec;    
    init_time += t2-t1;    
    
    return; 
}


void Solver::lie_trotter(const double current_time, const int task_id,
                            boost_array3d_t& solution, BrainModel& brain) {

    int ind = 0;
    int iter = 0;
    int i, j, k, i_loc;
    double new_val;

    /* compute norms */
    double error   = 1.0;
    double t_diff  = 0.0;

    int is, ie;
    std::tie(is, ie) = this->dims.local_i;
    
    /* auxiliary matrices for parallel processes data exchange */
    boost_array2d_t before(boost::extents[this->ny][this->nz]);
    boost_array2d_t after(boost::extents[this->ny][this->nz]);

    /* auxiliary variable to store intermediate solution of Jacobi method */
    boost_array3d_t C_iter(boost::extents[this->dims.nx_local][this->ny][this->nz]);
    boost_array3d_t C_iter_old(boost::extents[this->dims.nx_local][this->ny][this->nz]);
  
    /* Lie-Trotter formula: linear component solved with Jacobi */
    C_iter_old = solution;
         
    while(iter < ITER_MAX && error >= CONV_UP && error <= CONV_DO ) {
      
        std::fill_n(before.data(), before.num_elements(), 0.0);        
        std::fill_n(after.data(), after.num_elements(), 0.0);
    
        t_diff = 0.0;
    
        t1 = 0.0; t2 = 0.0;
        t_elap = gettimeofday(&tp, NULL);
        t1     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
        
# if defined _MPI        
        prepare_solution(task_id,num_tasks,nx_loc,ny,nz,&C_iter_old,&before,&after);
# endif

        #pragma omp parallel for default(shared) private(i,j,k,new_val,ind,i_loc) \
        schedule(static) collapse(3) reduction(+:t_diff)
        for(i = is; i < ie; i++) {      
            for(j = 1; j < ny-1; j++) {
                for(k = 1; k < nz-1; k++) {

                    ind = (i-is)*ny*nz + j*nz + k;
                    i_loc = i - is; 
                    new_val = 0.0;
	  
                    /* fixed-boundary conditions */
	                if(i > 0 && i < nx-1) { 
	  
                        # if defined _MPI 

                        if(i == is) {
                            new_val += C_iter_old.get(i_loc+1,j,k)*(*coeffs)[ind][1] + 
                                before.get(j,k)*(*coeffs)[ind][2];
                        }
                        else if(i == ie-1) {
                            new_val += after.get(j,k)*(*coeffs)[ind][1] + 
                                C_iter_old.get(i_loc-1,j,k)*(*coeffs)[ind][2];
                        }
                        else {
                            new_val += C_iter_old.get(i_loc+1,j,k)*(*coeffs)[ind][1] + 
                                C_iter_old.get(i_loc-1,j,k)*(*coeffs)[ind][2];
                        }
                        
                        new_val += C->get(i_loc,j,k) +
                                C_iter_old.get(i_loc,j+1,k) * (*coeffs)[ind][3] + 
                                C_iter_old.get(i_loc,j-1,k) * (*coeffs)[ind][4] +
                                C_iter_old.get(i_loc,j,k+1) * (*coeffs)[ind][5] + 
                                C_iter_old.get(i_loc,j,k-1) * (*coeffs)[ind][6];
                                
                        new_val /= (*coeffs)[ind][0];
                        C_iter.set(i_loc,j,k,new_val);
                        t_diff += pow(C_iter.get(i_loc,j,k) - C_iter_old.get(i_loc,j,k),2);
                        # else

                        new_val = ( solution[i][j][k] +
                            C_iter_old[i+1][j][k] * (*coeffs)[ind][1] +
                            C_iter_old[i-1][j][k] * (*coeffs)[ind][2] +
                            C_iter_old[i][j+1][k] * (*coeffs)[ind][3] + 
                            C_iter_old[i][j-1][k] * (*coeffs)[ind][4] +
                            C_iter_old[i][j][k+1] * (*coeffs)[ind][5] + 
                            C_iter_old[i][j][k-1] * (*coeffs)[ind][6] );
                        new_val /= (*coeffs)[ind][0];

                        C_iter[i][j][k] = new_val;
                        t_diff += pow(C_iter[i][j][k] - C_iter_old[i][j][k],2);

                        # endif
                                
                    }
                }
            }
        }
                  
        # if defined _MPI
        MPI_Allreduce(&t_diff, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        error = sqrt(error);
        # else
        error = sqrt(t_diff);
        # endif

        C_iter_old = C_iter;
        ++iter;
    }

    t_elap = gettimeofday(&tp, NULL);
    t2     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
    linear_time += t2 - t1;
        
    /* Lie-Trotter formula: non-linear component */

    C_iter_old = C_iter;
    std::fill_n(C_iter.data(), C_iter.num_elements(), 0.0);
    
    t1 = 0.0; t2 = 0.0;
    t_elap = gettimeofday(&tp, NULL);
    t1     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
    
    double rho_;

    #pragma omp parallel for default(shared) private(i,j,k,rho_,i_loc,new_val) \
    schedule(static) collapse(3)
    for(i = is; i < ie; i++) {      
        for(j = 1; j < ny-1; j++) {
            for(k = 1; k < nz-1; k++) {

                i_loc = i - is;
    
                if( (*brain.diffusion)[i][j][k] > 0.0 ) {
                    if(current_time < brain.t_radio) {
                        rho_ = brain.rho;
                    }
                    else {
                        rho_ = brain.rho - 1.0 + brain.s;
                    }
                } 
                else {
                    rho_ = 0.0;  
                }
        
                if(i > 0 && i < nx-1) { 
# if defined _MPI
                    if( C_iter_old.get(i_loc,j,k) > 0.0 ) {
                        new_val = exp( log(C_iter_old.get(i_loc,j,k) ) * exp(-1.0*rho_*dT) );
                        C_iter.set(i_loc,j,k,new_val);	      
                    }
# else
                    if( C_iter_old[i][j][k] > 0.0 ) {
                        new_val = exp( log(C_iter_old[i][j][k] ) * exp(-1.0*rho_*dT) );
                        C_iter[i_loc][j][k] = new_val;	      
                    }
# endif    
                }
            }
        }
    } 
    
    t_elap = gettimeofday(&tp, NULL);
    t2     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
    nonlinear_time += t2 - t1;
        
    solution = C_iter;
}

void Solver::expl(const int task_id, const int num_tasks,
			 boost_array3d_t& solution, BrainModel& brain) {

    int ind = 0;
    int i,j,k,i_loc;
    double new_val;
    double eps = 0.00001;
    int is, ie;
    std::tie(is, ie) = this->dims.local_i;

    /* auxiliary matrices for parallel processes data exchange */
    boost_array2d_t before(boost::extents[this->ny][this->nz]);
    boost_array2d_t after(boost::extents[this->ny][this->nz]);

    /* auxiliary variable to store intermediate solution of Jacobi method */
    boost_array3d_t C_iter(boost::extents[this->dims.nx_local][this->ny][this->nz]);
    boost_array3d_t C_iter_old(boost::extents[this->dims.nx_local][this->ny][this->nz]);
  
    C_iter_old = solution;
                    
    std::fill_n(before.data(), before.num_elements(), 0.0);        
    std::fill_n(after.data(), after.num_elements(), 0.0);
    
# if defined _MPI
    prepare_solution(task_id,num_tasks,this->dims.nx_local,ny,nz,&C_iter_old,&before,&after);
# endif

    #pragma omp parallel for default(shared) private(i,j,k,new_val,ind,i_loc) \
    schedule(static) collapse(3)
    for(i = is; i < ie; i++) {
        for(j = 1; j < ny-1; j++) {
            for(k = 1; k < nz-1; k++) {
        
                ind = (i-is)*ny*nz + j*nz + k;
                i_loc = i - is; 
        
                if(i > 0 && i < nx-1) { /* afuera de los bordes */
	  
	                new_val = 0.0;
	  
# if defined _MPI
            
	                /* version paralela */
	    
                    if(i == is) {
	                    new_val += C_iter_old[i_loc+1][j][k] * (*coeffs)[ind][1] + 
	                        before[j][k] * (*coeffs)[ind][2];
	                }
	                else if(i == ie-1) {
	                    new_val += after[j][k] * (*coeffs)[ind][1] + 
	                        C_iter_old[i_loc-1][j][k] * (*coeffs)[ind][2];
	                }
	                else {
	                    new_val += C_iter_old[i_loc+1][j][k] * (*coeffs)[ind][1] + 
	                        C_iter_old[i_loc-1][j][k] * (*coeffs)[ind][2];
	                }

                    new_val += C_iter_old[i_loc][j+1][k] * (*coeffs)[ind][3] + 
                                C_iter_old[i_loc][j-1][k] * (*coeffs)[ind][4] +
                                C_iter_old[i_loc][j][k+1] * (*coeffs)[ind][5] + 
                                C_iter_old[i_loc][j][k-1] * (*coeffs)[ind][6];
	  
                    new_val += C_iter_old[i_loc][j][k] * (*coeffs)[ind][0] - 
                        C_iter_old[i_loc][j][k] * log( sqrt( pow(C_iter_old[i_loc][j][k] , 2) + eps ) ) 
                        * (*coeffs)[ind][7];  	    
# else  
                    /* version scalar */
                    new_val = ( solution[i][j][k] +
                        C_iter_old[i+1][j][k] * (*coeffs)[ind][1] +
                        C_iter_old[i-1][j][k] * (*coeffs)[ind][2] +
                        C_iter_old[i][j+1][k] * (*coeffs)[ind][3] + 
                        C_iter_old[i][j-1][k] * (*coeffs)[ind][4] +
                        C_iter_old[i][j][k+1] * (*coeffs)[ind][5] + 
                        C_iter_old[i][j][k-1] * (*coeffs)[ind][6] );

                    new_val += C_iter_old[i][j][k] * (*coeffs)[ind][0] -
                        C_iter_old[i][j][k] * log( sqrt( pow(C_iter_old[i][j][k], 2) + eps ) ) 
                        * (*coeffs)[ind][7];    
# endif
                    /* update the matrix element with the new solution */
                    C_iter[i_loc][j][k] = new_val;
	            }
	        }
        }
    }

    solution = C_iter;
}


void Solver::prepare_solution(const int task_id, const int num_tasks,
				   boost_array3d_t& solution, boost_array2d_t& before, boost_array2d_t& after) {

# if defined _MPI
  
    int i,j,k;
    int ierr;
    
    double *t_mat1; 
    double *t_mat2;
    
    MPI_Status status;
    
    /* Para mandar:
     * M1 = mando C(task_id*nx_local,:,:) a la procesadora precedente 
     * M2 = mando C( (task_id+1)*nx_local-1,:,:) a la procesadora siguiente
     * Para recibir:
     * R1 = recibo t_mat1 de la procesadora siguiente y lo pongo en *after
     * R2 = recibo t_mat2 de la procesadora precedente y lo pongo en *before
     */
    
    t_mat1 = new double[ny*nz];
    t_mat2 = new double[ny*nz];
    
    /* llenar el *after */
    if(task_id > 0) {
      int i = 0;
      for(int j = 0; j < ny; j++) {
	for(int k = 0; k < nz; k++) {
	  t_mat1[k+nz*j] = C->get(i,j,k);
	}
      }
      ierr = MPI_Send(t_mat1,ny*nz,MPI_DOUBLE,task_id-1,ATAG,MPI_COMM_WORLD);
    }
    
    if(task_id < num_tasks - 1) {
      ierr = MPI_Recv(t_mat1,ny*nz,MPI_DOUBLE,task_id+1,ATAG,MPI_COMM_WORLD,&status); 
      after->copy(t_mat1);
    }
    
    /* llenar el *before */
    if(task_id < num_tasks - 1) {
      int i = nx_local - 1;
      for(int j = 0; j < ny; j++)
	for(int k = 0; k < nz; k++)
	  t_mat2[k+nz*j] = C->get(i,j,k);
      ierr = MPI_Send(t_mat2,ny*nz,MPI_DOUBLE,task_id+1,BTAG,MPI_COMM_WORLD);
    }
    
    if(task_id > 0) {
      ierr = MPI_Recv(t_mat2,ny*nz,MPI_DOUBLE,task_id-1,BTAG,MPI_COMM_WORLD,&status); 
      before->copy(t_mat2);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    delete[] t_mat1;
    delete[] t_mat2;
    
# endif
    
    return; 
}


