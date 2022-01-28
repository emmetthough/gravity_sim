ROTARY ATTRACTOR GRAVITY SIM
============================

Simulations and analysis as part of the rotary attractor project with the Gratta Gravity Group at Stanford University, by Emmett Hough.

To install, noting the path to the src since no wheels exist on PyPI, use:
	
	pip install ./gravity_sim

If you intend to edit doe and want the import calls to reflect those changes, install in developer mode:

	pip install -e gravity_sim

---------------------------

'lib' contains executables meant to be installed and imported. holes_analysis contains functions for analysis of the simulation data produced by the scripts in the simulations directory. attractor_profile and symmertric_attractor_profile contain all the working of the FEM simulation itself.

'simulations' contains scripts that handle parallelization, execution, and saving of simulation data with different spatial configurations and density profiles.

'notebooks' contain examples of how one would use the analysis scripts to read in data from a path and produce figures using the various functions in holes_analysis. The data itself has not been included in the repo due to its large size and thus you'd have to change the path to data generated yourself.

All scripts should be commented well enough to understand what each function does. To create simulation data of your own, follow the structure in any of the 'simulation' scripts, changing the density profile and simulation parameters as necessary, as well as specifiying the path to save the data in. I think this root directory needs to exist prior to the simulation running as an empty dir. Follow the example notebooks to see how to use the holes_analysis functions with generated data.
