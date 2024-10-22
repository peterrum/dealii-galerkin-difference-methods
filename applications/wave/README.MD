

```bash
mkdir matrices

# without cut
for i in `seq 20 20 200`; do ./applications/wave/wave-ev --n_subdivisions $i --compute_kappa_m --write_m --write_s --radius 1.4 --file_prefix matrices/matrix_$i; done

# with cut (but without stabilization)
for i in `seq 20 20 200`; do ./applications/wave/wave-ev --n_subdivisions $i --compute_kappa_m --write_m --write_s --disable_ghost_penalty --file_prefix matrices/matrix_cut_$i; done

# with cut
for i in `seq 20 20 200`; do ./applications/wave/wave-ev --n_subdivisions $i --compute_kappa_m --write_m --write_s --file_prefix matrices/matrix_cut_stab_$i; done

tar cvf matrices.tar matrices

```

run experiments with different alpha:
```bash
./applications/wave/wave-ev --n_subdivisions 100 --compute_kappa_m --compute_kappa_s --radius 1.0 --alpha 0.1
./applications/wave/wave-ev --n_subdivisions 100 --compute_kappa_m --compute_kappa_s --radius 1.0 --alpha 0.1 --disable_ghost_penalty
```