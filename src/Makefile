CPPS = g++ 
CPP  = mpic++

FLAGS = -O3 -std=c++17 -ffast-math -ftree-vectorize -funroll-loops -I ${BOOST_ROOT} 
FLAGO = $(FLAGS) -D_OMP -fopenmp
FLAGP = $(FLAGO) -D_MPI 
FLAGM = $(FLAGS) -D_MPI

TUMOR_SRC = brain.cpp solver.cpp main.cpp
TUMOR_OBJ = brain.o solver.o main.o

# serial version
tumor-serial.o: $(TUMOR_SRC) 
	$(CPPS) $(FLAGS) -c $(TUMOR_SRC) 

serial: tumor-serial.o
	$(CPPS) $(FLAGS) $(TUMOR_OBJ) -o tumor-serial.x	                          

# serial version with OpenMP
tumor-omp.o: $(TUMOR_SRC) 
	$(CPPS) $(FLAGO) -c $(TUMOR_SRC) 

omp: tumor-omp.o     
	$(CPPS) $(FLAGO) $(TUMOR_OBJ) -o tumor-omp.x

# Parallel MPI version
tumor-mpi.o: $(TUMOR_SRC) 
	$(CPP) $(FLAGM) -c $(TUMOR_SRC)

mpi: tumor-mpi.o
	$(CPP) $(FLAGM) $(TUMOR_OBJ) -o tumor-mpi.x


# Parallel MPI+OpenMP version
tumor-hybrid.o: $(TUMOR_SRC) 
	$(CPP) $(FLAGP) -c $(TUMOR_SRC)

hybrid: tumor-hybrid.o
	$(CPP) $(FLAGP) $(TUMOR_OBJ) -o tumor-hybrid.x

all:
	make serial
	make omp
	make mpi
	make hybrid

# remove all compilation objects
clean:
	rm -f *.o tumor-serial.x tumor-omp.x tumor-mpi.x tumor-hybrid.x
