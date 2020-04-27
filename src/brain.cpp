#include <boost/algorithm/string.hpp>
#include <boost/multi_array.hpp>

# include "brain.hpp"


BrainModel::BrainModel() {
  
 /* Valores predefinidos del modelo */
 Cm      = 1.*pow(10,8);
 rho     = 0.107; 
 D_white = 0.255;
 D_grey  = 0.051;
 dose    = 1.8;
 alpha   = 0.036;
 beta    = 0.0036;
 s = exp( -alpha*dose -beta*pow(dose,2) );
 t_radio = 15.0;

 /* define the initial position and size of the tumor */
# if defined TEST
 tumor_size = 5;
 tumor_position[0] = 3; 
 tumor_position[1] = 3; 
 tumor_position[2] = 3;
 fname = "diffusion_test.csv";
# else
 tumor_size = 30;
 tumor_position[0] = 30; 
 tumor_position[1] = 93; 
 tumor_position[2] = 30;
 fname = "diffusion.csv";
# endif
}
  
BrainModel::BrainModel(const double Cm, 
                       const double rho, 
	                     const double D_white, 
                       const double D_grey,
	                     const double alpha, 
                       const double beta, 
                       const double dose,
	                     const int tumor_size, 
                       const double t_radio) 
{
    this->Cm = Cm; 
    this->rho = rho;
    this->D_white = D_white;
    this->D_grey = D_grey;
    this->dose = dose;
    this->alpha = alpha;
    this->beta = beta;
    this->s = exp( -alpha*dose -beta*pow(dose,2) );
    this->t_radio = t_radio;

    this->tumor_size = tumor_size;
    tumor_position[0] = 30; 
    tumor_position[1] = 93; 
    tumor_position[2] = 30;

}
  
/* read input diffusion coefficients and populates current brain model */
void BrainModel::read( matrix3D *D, 
                       int &nx, 
                       int &ny, 
                       int &nz, 
                       const int my_id) 
{
  
    std::ifstream f;
    std::string line;

    nx = 0; 
    ny = 0; 
    nz = 0;
    std::vector<double> t_D; 

    if( my_id == 0) {

        f.open(fname, std::ifstream::in);
        if( !f.is_open() ) {
        std::cout << " Error opening input file " << std::endl;
        std::cout << " The program will not work! " << std::endl;
        # if defined _MPI
            MPI_Finalize();
        # endif
            exit(EXIT_FAILURE);        
        }
            
        std::getline(f, line);  /* read the heading */
        while( std::getline(f, line) ) {        

            std::vector<std::string> result;
            boost::algorithm::split(result, line, boost::is_any_of(","));

            nx = max(nx, std::stoi(result[0]));
            ny = max(ny, std::stoi(result[1]));
            nz = max(nz, std::stoi(result[2]));
            int D_read = std::stoi(result[3]);

            if(D_read >= this->is_white.first && D_read <= this->is_white.second) {
                t_D.push_back(D_white);
            }
            else if(D_read >= this->is_grey.first && D_read < this->is_grey.second) {
                t_D.push_back(D_grey);
            }
            else {
                t_D.push_back(0.0);
            }

        }
        f.close();
    }  
  
  /* creacion del vector de coeficientes de difusion para todos los procecos */
    D->reset_dimension(nx,ny,nz);
  
    int ind = 0;
    if(my_id == 0) {
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny; j++) {
                for(int k = 0; k < nz; k++) {
                    D->set(i,j,k,t_D[ind]);
                    ++ind;
                }
            }
        }
    }
  
    return;
  
 }
  
 void BrainModel::init_tumor(const int task_id, matrix3D *C, const int is, const int ie) {
   
   int i,j,k;
   double init_value = Cm/2.0;
   int i_start = tumor_position[0];
   int j_start = tumor_position[1];
   int k_start = tumor_position[2];
   int dim     = tumor_size;
   double radius = dim/2.0;
      
   for(i = is; i < ie; i++) {
     int i_loc = i - is;
     for(j = j_start - radius ;j <= j_start + radius; j++) {
       for(k = k_start - radius;k <= k_start + radius; k++) {

	 double ix = (double) i-i_start;
	 double iy = (double) j-j_start;
         double iz = (double) k-k_start;

	 if(i >= i_start - radius && i <= i_start + radius) {     
	   	   
	    if( sqrt( pow(ix,2) + pow(iy,2) + pow(iz,2) ) < radius )
	      C->set(i_loc,j,k,init_value);
         }
       }
     }
   }
      
   return;
 }
 
 double BrainModel::norm_tumor(const int is, const int ie, matrix3D *C) {
   
   int i,j,k;
   double init_value = Cm/2.0;
   int i_start = tumor_position[0];
   int j_start = tumor_position[1];
   int k_start = tumor_position[2];
   int dim     = tumor_size;
   double radius = dim/2.0;
   
   double norm = 0.0;   
   
   for(i = is; i < ie; i++) {
     int i_loc = i - is;
     for(j = j_start - radius ;j <= j_start + radius; j++) {
       for(k = k_start - radius;k <= k_start + radius; k++) {

	 double ix = (double) i-i_start;
	 double iy = (double) j-j_start;
         double iz = (double) k-k_start;

	 if(i >= i_start - 2.0 && i <= i_start + 2.0) {     
	   	   
	    if( sqrt( pow(ix,2) + pow(iy,2) + pow(iz,2) ) < 2.0 )
	      norm += pow(C->get(i_loc,j,k),2);
         }
       }
     }
   }   
   norm = sqrt(norm);
   return norm; 
 }