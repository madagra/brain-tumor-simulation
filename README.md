# Simulation of brain tumor evolution
 
Simulation of the spread of a tumor in the brain using simple diffusion equations.

# Running the solver

The solver needs to have install `g++7` or more recent and `cmake` version 3.13 or more. In order to compile use:

```
mkdir build
cd build
cmake ..
make
```

This creates the executable `run_simulation` for running the solver. When running it, you will be prompted with several input parameters:

* Spacial step and temporal step: size of one step of the solver grid (use 1.0 if not known) and the size of one time step in the simulation.

* Total simulation time: how long the simulation must run. The number of
diffusion equation solutions computed is determined by *total simulation time/temporal step*. If you wants to see the full tumor diffusion set a total time greater than 75.

* Every which number of steps printing the partial solution for visualization.

* The time step when the radiotherapy effect is activated for stopping the
tumor growth.

* Method used to solve the diffusion equation. Select 0 for the Lie-Trotter
method and 1 for the direct explicit method.

# Visualize tumor evolution

The `visualization` folder contains a simple web application built using the [Dash](https://plotly.com/dash/) library. This allows to
visualize the tumor evolution based on the solution of the diffusion equations.

Using your favorite virtual environment tool, install the requirements provided. You can use `pipenv` as follows:

```
cd visualization
pipenv shell
pipenv install
```

To run the application within the virtual environment, and copy inside the folder `solution` created after
running the solver and run the application:

```
cd visualization
cp -r ../build/solution .
python plot_evolution.py
```

You can now open it within you browser.