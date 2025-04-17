rm *.evout


for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_3_0.evout
  ./advection-ev --compute_kappa_m --fe_degree 3 --n_subdivisions $i --n_ghost_cells 0 | tee -a m_ev_3_0.evout
done

for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_3_1.evout
  ./advection-ev --compute_kappa_m --fe_degree 3 --n_subdivisions $i --n_ghost_cells 1| tee -a m_ev_3_1.evout
done

for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_3_2.evout
  ./advection-ev --compute_kappa_m --fe_degree 3 --n_subdivisions $i --n_ghost_cells 2| tee -a m_ev_3_2.evout
done

for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_3_3.evout
  ./advection-ev --compute_kappa_m --fe_degree 3 --n_subdivisions $i --n_ghost_cells 2 --disable_ghost_penalty | tee -a m_ev_3_3.evout
done




for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_5_0.evout 
  ./advection-ev --compute_kappa_m --fe_degree 5 --n_subdivisions $i --n_ghost_cells 0 | tee -a m_ev_5_0.evout
done

for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_5_1.evout
  ./advection-ev --compute_kappa_m --fe_degree 5 --n_subdivisions $i --n_ghost_cells 1| tee -a m_ev_5_1.evout
done

for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_5_2.evout
  ./advection-ev --compute_kappa_m --fe_degree 5 --n_subdivisions $i --n_ghost_cells 2| tee -a m_ev_5_2.evout
done

for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_5_3.evout
  ./advection-ev --compute_kappa_m --fe_degree 5 --n_subdivisions $i --n_ghost_cells 3| tee -a m_ev_5_3.evout
done

for ((i=8;i<=128;i=i*2)); do
  echo "number of cells: "$i | tee -a m_ev_5_4.evout
  ./advection-ev --compute_kappa_m --fe_degree 5 --n_subdivisions $i --n_ghost_cells 3 --disable_ghost_penalty | tee -a m_ev_5_4.evout
done




for i in *.evout; do
  cat $i | grep "number of cells:" | awk -F ':' '{print $2}' >| temp0.out
  cat $i | grep "condition number:" | awk -F ':' '{print $2}' >| temp1.out
  paste -d, temp0.out temp1.out >| $i
done
