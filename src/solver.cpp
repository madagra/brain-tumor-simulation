# include "solver.h"

classSystem::classSystem() {

  nx      = 181;
  ny      = 217;
  nz      = 181;
  dim_tot = nx*ny*nz;
  
  /* asumiendo 4 procesadoras por default */
  nx_loc  = nx/4;
  nx_rest = nx%4;
  
  /* coefficientes de la matriz de Jacobi */
  dim_coeffs = nx_loc*ny*nz;
  coeffs = new double*[dim_coeffs]; 
  for(int i = 0; i < dim_coeffs; i++) {
    coeffs[i] = new double[s_order+1];
    for( int j = 0; j <= s_order; j++) {
      coeffs[i][j] = 0.0;
    }
  }
  
  init_time = 0.0;
  linear_time = 0.0;
  nonlinear_time = 0.0;

  return;
}

classSystem::classSystem(const int nx_, const int ny_, const int nz_, 
			 const int num_tasks, const int task_id, 
			 const double dX_, const double dT_) {
 
  nx = nx_;
  ny = ny_;
  nz = nz_;
  
  dT = dT_;
  dX = dX_;

  /* definicion de las dimensiones por cada proceso */
  dim_tot  = nx*ny*nz;
  dim_rest = (nx%num_tasks)*ny*nz;
  nx_rest  = nx%num_tasks;
  nx_loc   = nx/num_tasks;
  
  /* definicion de los indices locales de la grilla */
# if defined _MPI
  if(task_id < num_tasks - 1) {
    is =  task_id*nx_loc;
    ie = (task_id+1)*nx_loc;
  }
  else {
    is = task_id*nx_loc;
    ie = nx;
    nx_loc += nx_rest;
  }
# else
  is = 0;
  ie = nx;
# endif
  
  /* coefficientes de la matriz de Jacobi */
  s_order = 7;
  dim_coeffs = nx_loc*ny*nz;
  coeffs = new double*[dim_coeffs]; 
  for(int i = 0; i < dim_coeffs; i++) {
    coeffs[i] = new double[s_order+1];
    for( int j = 0; j <= s_order; j++) {
      coeffs[i][j] = 0.0;
    }
  }
  
  init_time = 0.0;
  linear_time = 0.0;
  nonlinear_time = 0.0;
  
  return;
  
}

classSystem::~classSystem() {
  
  for(int i = 0; i < dim_coeffs; i++)
    delete[] coeffs[i]; 
  
  return;
  
}

void classSystem::start(matrix3D *D, BrainModel *brain, const int method) {
  
    int i,j,k;
    double D_loc;
    int ind = 0;
    double t_coeff = dT / (2.0 * pow(dX,2) );
    double rho_;

    t_elap = gettimeofday(&tp, NULL);
    t1     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
          
    for(i = is; i < ie; i++) {
      for(j = 1; j < ny-1; j++) {
        for(k = 1; k < nz-1; k++) {    
	  
	  D_loc = D->get(i,j,k);
          ind = (i-is)*ny*nz + j*nz + k;

	  if( D_loc > 0.0 ) {
	    rho_ = brain->rho + brain->s - 1.0;
	  } 
          else {
	    rho_ = 0.0;  
          }
	  
	  if( i > 0 && i < nx - 1) {	

            coeffs[ind][1] = t_coeff * ( D->get(i+1,j,k) + D_loc );
	    coeffs[ind][2] = t_coeff * ( D->get(i-1,j,k) + D_loc );
            coeffs[ind][3] = t_coeff * ( D->get(i,j+1,k) + D_loc );
	    coeffs[ind][4] = t_coeff * ( D->get(i,j-1,k) + D_loc );
            coeffs[ind][5] = t_coeff * ( D->get(i,j,k+1) + D_loc );
	    coeffs[ind][6] = t_coeff * ( D->get(i,j,k-1) + D_loc );
	    
	    if(method == 0) {
	      ++coeffs[ind][0]; /* 1.0 + */
	      for(int jj = 1; jj < 7; jj++)
	        coeffs[ind][0] += coeffs[ind][jj]; /* todos los otros coeficientes */
	      }
	    else if(method == 1) {
	      ++coeffs[ind][0]; /* 1.0 + */
	      for(int jj = 1; jj < 7; jj++)
	        coeffs[ind][0] = coeffs[ind][0] - coeffs[ind][jj]; /* todos los otros coeficientes */
	      coeffs[ind][7] = dT * rho_;
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


void classSystem::lie_trotter(const double TIME, const int task_id, const int num_tasks,
			      matrix3D *D, matrix3D *C, BrainModel *brain) {

  int ind = 0;
  int iter = 0;
  int i,j,k,i_loc;
  double new_val;

  /* calculo de las normas */
  double error   = 1.0;
  double t_diff  = 0.0;
  
  /* matrizes para send/receives */
  matrix2D before(ny,nz);
  matrix2D after(ny,nz);
  
  /* solucion en iteracion de Jacobi */
  matrix3D C_iter(nx_loc,ny,nz);
  matrix3D C_iter_old(nx_loc,ny,nz);
  
  /* Lie-Trotter formula: parte lineal con Jacobi */
  C_iter_old.copy(C);
          
  while(iter < ITER_MAX && error >= CONV_UP && error <= CONV_DO ) {
      
    after.init(0.0);
    before.init(0.0);
    t_diff = 0.0;
 
    t1 = 0.0; t2 = 0.0;
    t_elap = gettimeofday(&tp, NULL);
    t1     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
    
    prepare_solution(task_id,num_tasks,nx_loc,ny,nz,&C_iter_old,&before,&after);

# if defined _OMP
    #pragma omp parallel for default(shared) private(i,j,k,new_val,ind,i_loc) \
    schedule(static) collapse(3) reduction(+:t_diff)
# endif      
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
	      new_val += C_iter_old.get(i_loc+1,j,k)*coeffs[ind][1] + 
	      before.get(j,k)*coeffs[ind][2];
	    }
	    else if(i == ie-1) {
	      new_val += after.get(j,k)*coeffs[ind][1] + 
	      C_iter_old.get(i_loc-1,j,k)*coeffs[ind][2];
	    }
	    else {
	      new_val += C_iter_old.get(i_loc+1,j,k)*coeffs[ind][1] + 
	      C_iter_old.get(i_loc-1,j,k)*coeffs[ind][2];
	    }
	    
	    new_val += C->get(i_loc,j,k) +
	               C_iter_old.get(i_loc,j+1,k) * coeffs[ind][3] + 
	               C_iter_old.get(i_loc,j-1,k) * coeffs[ind][4] +
	               C_iter_old.get(i_loc,j,k+1) * coeffs[ind][5] + 
	               C_iter_old.get(i_loc,j,k-1) * coeffs[ind][6];
	    	    
	    new_val = new_val/coeffs[ind][0];
            C_iter.set(i_loc,j,k,new_val);
	    t_diff += pow(C_iter.get(i_loc,j,k) - C_iter_old.get(i_loc,j,k),2);
	      
# else  
	    /* version scalar */
	    
    	    new_val = ( C->get(i,j,k) +
	                C_iter_old.get(i+1,j,k) * coeffs[ind][1] +
	                C_iter_old.get(i-1,j,k) * coeffs[ind][2] +
		        C_iter_old.get(i,j+1,k) * coeffs[ind][3] + 
		        C_iter_old.get(i,j-1,k) * coeffs[ind][4] +
                        C_iter_old.get(i,j,k+1) * coeffs[ind][5] + 
                        C_iter_old.get(i,j,k-1) * coeffs[ind][6] ) / 
                        coeffs[ind][0];
	    	    
            C_iter.set(i,j,k,new_val);
	    t_diff += pow(C_iter.get(i,j,k) - C_iter_old.get(i,j,k),2);

# endif
	    
	    }
	  }
        }
      }
                  
# if defined _MPI
      MPI_Allreduce(&t_diff,&error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      error = sqrt(error);
# else
      error = sqrt(t_diff);
# endif    
      C_iter_old.copy(&C_iter);
      ++iter;

    }

    t_elap = gettimeofday(&tp, NULL);
    t2     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
    linear_time += t2 - t1;
        
    /* Lie-Trotter formula: parte non lineal */
    C_iter_old.copy(&C_iter);
    C_iter.init(0.0);
    
    t1 = 0.0; t2 = 0.0;
    t_elap = gettimeofday(&tp, NULL);
    t1     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
    
    double rho_;
# if defined _OMP
    #pragma omp parallel for default(shared) private(i,j,k,rho_,i_loc,new_val) \
    schedule(static) collapse(3)
# endif    
    for(i = is; i < ie; i++) {      
      for(j = 1; j < ny-1; j++) {
        for(k = 1; k < nz-1; k++) {

	  i_loc = i - is;
	  
	  if( D->get(i,j,k) > 0.0 ) {
	    if(TIME < brain->t_radio) {
              rho_ = brain->rho;
	    }
	    else {
	      rho_ = brain->rho - 1.0 + brain->s;
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
	    if( C_iter_old.get(i,j,k) > 0.0 ) {
	       new_val = exp( log(C_iter_old.get(i,j,k) ) * exp(-1.0*rho_*dT) );
	       C_iter.set(i,j,k,new_val);	      
	    }
# endif
	    
	  }
	}
      }
    } 
    
    t_elap = gettimeofday(&tp, NULL);
    t2     = (double) tp.tv_sec+(1.e-6)*tp.tv_usec;
    nonlinear_time += t2 - t1;
        
    C->copy(&C_iter);
    
    return;
}

void classSystem::expl(const int task_id, const int num_tasks,
			 matrix3D *D, matrix3D *C, BrainModel *brain) {

  int ind = 0;
  int i,j,k,i_loc;
  double new_val;
  double eps = 0.00001;

  /* matrizes para send/receives */
  matrix2D before(ny,nz);
  matrix2D after(ny,nz);
  
  /* solucion en iteracion de Jacobi */
  matrix3D C_iter(nx_loc,ny,nz);
  matrix3D C_iter_old(nx_loc,ny,nz);
  
  C_iter_old.copy(C);
                
  after.init(0.0);
  before.init(0.0);
  
  prepare_solution(task_id,num_tasks,nx_loc,ny,nz,&C_iter_old,&before,&after);
  
# if defined _OMP
  #pragma omp parallel for default(shared) private(i,j,k,new_val,ind,i_loc) schedule(static) collapse(3)
# endif   
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
	    new_val += C_iter_old.get(i_loc+1,j,k)*coeffs[ind][1] + 
	    before.get(j,k)*coeffs[ind][2];
	  }
	  else if(i == ie-1) {
	    new_val += after.get(j,k)*coeffs[ind][1] + 
	    C_iter_old.get(i_loc-1,j,k)*coeffs[ind][2];
	  }
	  else {
	    new_val += C_iter_old.get(i_loc+1,j,k)*coeffs[ind][1] + 
	    C_iter_old.get(i_loc-1,j,k)*coeffs[ind][2];
	  }
	    
	  new_val += C_iter_old.get(i_loc,j+1,k) * coeffs[ind][3] + 
	             C_iter_old.get(i_loc,j-1,k) * coeffs[ind][4] +
	             C_iter_old.get(i_loc,j,k+1) * coeffs[ind][5] + 
	             C_iter_old.get(i_loc,j,k-1) * coeffs[ind][6];
	  
          new_val += C_iter_old.get(i_loc,j,k) * coeffs[ind][0] -
                     C_iter_old.get(i_loc,j,k) * log( sqrt( pow(C_iter_old.get(i_loc,j,k),2) + eps ) ) 
		     * coeffs[ind][7];  	    
		     
          C_iter.set(i_loc,j,k,new_val);
	      
# else  
	  /* version scalar */

	  new_val += C_iter_old.get(i+1,j,k) * coeffs[ind][1] + 
	             C_iter_old.get(i-1,j,k) * coeffs[ind][2] + 
	             C_iter_old.get(i,j+1,k) * coeffs[ind][3] + 
	             C_iter_old.get(i,j-1,k) * coeffs[ind][4] +
	             C_iter_old.get(i,j,k+1) * coeffs[ind][5] + 
	             C_iter_old.get(i,j,k-1) * coeffs[ind][6];
	  
          new_val += C_iter_old.get(i,j,k) * coeffs[ind][0] -
                     C_iter_old.get(i,j,k) * log( sqrt( pow(C_iter_old.get(i,j,k),2) + eps ) ) 
		     * coeffs[ind][7];  	    

          C_iter.set(i,j,k,new_val);

# endif	    
	  }
	}
      }
    }

    C->copy(&C_iter);
    
    return;
}


void classSystem::prepare_solution(const int task_id, const int num_tasks, 
				   const int nx_local, const int ny, const int nz, 
				   matrix3D *C, matrix2D *before, matrix2D *after) {

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


