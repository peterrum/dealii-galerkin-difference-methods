./advection-ev --compute_gev --fe_degree 3 --n_subdivisions 32 --n_ghost_cells 0 >|  spectrum_3_0.evout
./advection-ev --compute_gev --fe_degree 3 --n_subdivisions 32 --n_ghost_cells 3 >|  spectrum_3_3.evout

./advection-ev --compute_gev --fe_degree 5 --n_subdivisions 32 --n_ghost_cells 0 >|  spectrum_5_0.evout
./advection-ev --compute_gev --fe_degree 5 --n_subdivisions 32 --n_ghost_cells 4 >|  spectrum_5_4.evout