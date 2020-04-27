# include <iostream> 
# include <cstdlib>
# include <cmath>
# include <fstream>   
# include <sstream>   
# include <iomanip>
# include <string>

# if defined _MPI
# include <mpi.h>
# endif

# if not defined MATRIX_CLASS_H
# define MATRIX_CLASS_H

using namespace std;

class matrix3D {

private:
  int ii;
  int jj;
  int kk;
  double *my_mat;

public:

  matrix3D() {
   ii     = 0;
   jj     = 0;
   kk     = 0;
   my_mat = NULL;
  }
  
  matrix3D(int ii_, int jj_, int kk_) {
    ii = ii_;
    jj = jj_;
    kk = kk_;
    my_mat = new double[ii*jj*kk];
    init(0.0);    
  }
  
  void reset_dimension(int ii_, int jj_, int kk_) {
    ii = ii_;
    jj = jj_;
    kk = kk_;
    if(my_mat != NULL)
      delete [] my_mat;
    my_mat = new double[ii*jj*kk];
    init(0.0);    
  }
  
  void init(double value) {
    int dim_tot = ii*jj*kk;
    for(int i = 0; i < dim_tot; i++)
      my_mat[i] = value;
  }
  
  double get(int i, int j, int k) {
    return my_mat[i*jj*kk + j*kk + k];
  }
  
  void set(int i, int j, int k, double value) {
     my_mat[i*jj*kk + j*kk + k] = value;
  }
  
  void copy(matrix3D *mat) {
    int dim_tot = ii*jj*kk;
    for(int i = 0; i < dim_tot; i++)
      this->my_mat[i] = mat->my_mat[i];
  }
  
  void print(void) {
    int dim_tot = ii*jj*kk;
    for(int i = 0; i < dim_tot; i++)
      cout << i << " " << my_mat[i] << endl;
  }
  
  void fill(double *t_C) {
    int dim_tot = ii*jj*kk;
    for(int i = 0; i < dim_tot; i++)
      t_C[i] = this->my_mat[i];    
  }
  
  double norm(void) {
    double err = 0.0;
    for(int i = 0; i < ii; i++) {
      for(int j = 0; j < jj; j++) {
	for(int k = 0; k < kk; k++) {
	   err += pow(this->get(i,j,k),2);  
	}
      }
    }
    err = sqrt(err);
    return err;
  }
  
  double sum_elements(void) {
    double sum = 0.0;
    for(int i = 0; i < ii; i++) {
      for(int j = 0; j < jj; j++) {
	for(int k = 0; k < kk; k++) {
	   sum += this->get(i,j,k);  
	}
      }
    }
    return sum;
  }
  
  void print(string filename)
  {

    ofstream f;

    f.open(filename.c_str(),ofstream::out);
    f << "# vtk DataFile Version 2.0\n";
    f << "Comment goes here\n";
    f << "ASCII\n";
    f << endl;
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << ii << " " << jj << " " << kk << endl;
    f << endl;
    f << "ORIGIN  "  << 0.0 << " " << 0.0 << " " << 0.0 << endl;
    f << "SPACING "  << 1.0  << " " << 1.0  << " " << 1.0 << endl;
    f << endl;
    f << "POINT_DATA " << ii*jj*kk << endl;
    f << "SCALARS scalars float\n" ;
    f << "LOOKUP_TABLE default\n";
    f << endl;  
    for( int k = 0; k < kk; k++) {
       for( int j = 0; j < jj; j++) {
          for( int i = 0; i < ii; i++) {
            f << this->get(i,j,k) << endl;
          }
       }
    }
    f.close();
  }  
  
# if defined _MPI
  void bcast(MPI_Comm comm) {
    int dim_tot = ii*jj*kk;
    int ierr;
    ierr = MPI_Bcast( this->my_mat,dim_tot,MPI_DOUBLE,0,comm );
    if( ierr != MPI_SUCCESS) {
     cout << " ERROR broadcasting matrix3D. The program will stop! " << endl;
     MPI_Finalize();
     exit(EXIT_FAILURE);
    }
    return;
  }  
  
  void gather(double *C_tot, MPI_Comm comm ) {
    
    int ierr;
    int dim = ii*jj*kk;
    
    ierr = MPI_Gather(this->my_mat, dim , MPI_DOUBLE, C_tot, dim, MPI_DOUBLE, 0, comm);
    if( ierr != MPI_SUCCESS) {
     cout << " ERROR gathering matrix3D. The program will stop! " << endl;
     MPI_Finalize();
     exit(EXIT_FAILURE);
    }
    return;    
  }
# endif
  
  ~matrix3D(){
    if (my_mat!=NULL)
       delete [] my_mat;
    ii = 0;
    jj = 0;
    kk = 0;
  }

};

class matrix2D {

private:
  int ii;
  int jj;
  double *my_mat;

public:

  matrix2D() {
   ii     = 0;
   jj     = 0;
   my_mat = NULL;
  }
  
  matrix2D(int ii_, int jj_) {
    ii = ii_;
    jj = jj_;
    my_mat = new double[ii*jj];
    init(0.0);    
  }
  
  void init(double value) {
    int dim_tot = ii*jj;
    for(int i = 0; i < dim_tot; i++)
      my_mat[i] = value;
  }
  
  void reset_dimension(int ii_, int jj_) {
    ii = ii_;
    jj = jj_;
    if(my_mat != NULL)
      delete [] my_mat;
    my_mat = new double[ii*jj];
    init(0.0);    
  }
  
  double get(int i, int j) {
    return my_mat[i*jj + j];
  }
  
  void set(int i, int j, double value) {
     my_mat[i*jj + j] = value;
  }
  
  void info(void) {
    int dim_tot = ii*jj;
    cout << " ii/jj/dim_tot " << ii << " " << jj << " " << dim_tot << endl;
  }
  
  void print(void) {
    int dim_tot = ii*jj;
    for(int i = 0; i < dim_tot; i++)
      cout << i << " " << this->my_mat[i] << endl;
  }
   
  void copy(matrix2D *mat) {
    int dim_tot = ii*jj;
    double value;
    for(int i = 0; i < dim_tot; i++)
      this->my_mat[i] = mat->my_mat[i];
  }
  
  void copy(double *mat) {
    int dim_tot = ii*jj;
    double value;
    for(int i = 0; i < dim_tot; i++)
      this->my_mat[i] = mat[i];
  }
  
  void copy(matrix3D *mat, int it) {
    double value = 0.0;
    for(int i = 0; i < ii; i++) {
      for(int j = 0; j < jj; j++) {
	value = mat->get(it,i,j);
	this->set(i,j,value);
      }
    }
  }

  ~matrix2D(){
    if (my_mat!=NULL)
       delete [] my_mat;
    ii = 0;
    jj = 0;
  }

};

# endif