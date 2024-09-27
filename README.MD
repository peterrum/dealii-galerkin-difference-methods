### General

Build project:

```
git clone https://github.com/peterrum/dealii-galerkin-difference-methods.git
cd dealii-galerkin-difference-methods
mkdir build
cd build
cmake .. -DDEAL_II_DIR=/PATH/TO/THE/DEAL/II/BUILD/FOLDER
make -j 4
```

Switch to release mode:

```
make release
```

Switch back to debug mode:

```
make debug
```

Run wave application:

```
./applications/wave/wave-app 1 wave
```

Run wave application with MPI:


```
mpirun -np 5 ./applications/wave/wave-app 1 wave
```

Run code to estimate eigenvalues:

```
./applications/wave/wave-ev
```