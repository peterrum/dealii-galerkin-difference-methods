// Test Poisson problem: parallel, with preconditioner, 1D, 2D.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <gdm/data_out.h>
#include <gdm/system.h>

#include <fstream>



template <int dim>
void
test()
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  const unsigned int n_subdivisions   = 20;
  const unsigned int fe_degree        = 3;
  const unsigned int n_components     = 1;
  const unsigned int fe_degree_output = 2;

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  // Create GDM system
  GDM::System<dim> system(MPI_COMM_WORLD, fe_degree, n_components);

  // Create mesh
  if (true)
    {
      system.subdivided_hyper_cube(n_subdivisions, 0.0, +1.0);
    }
  else
    {
      const std::vector<unsigned int> repetitions = {{20, 10}};
      const Point<dim>                p1(0.0, 0.0);
      const Point<dim>                p2(2.0, 1.0);
      system.subdivided_hyper_rectangle(repetitions, p1, p2);
    }


  // Create finite elements
  const auto &fe = system.get_fe();

  // Create mapping
  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());

  // Create quadrature
  hp::QCollection<dim> quadrature;
  quadrature.push_back(QGauss<dim>(fe_degree + 1));

  // Create constraints
  AffineConstraints<Number> constraints(system.locally_active_dofs());
  system.make_zero_boundary_constraints(constraints);
  constraints.close();

  // Categorize cells
  system.categorize();

  // Create sparsity pattern and allocate sparse matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern(
    system.locally_owned_dofs(), MPI_COMM_WORLD);
  system.create_sparsity_pattern(constraints, sparsity_pattern);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  // create vectors
  const auto partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(system.locally_owned_dofs(),
                                                  system.locally_relevant_dofs(
                                                    constraints),
                                                  MPI_COMM_WORLD);

  VectorType rhs(partitioner);
  VectorType solution(partitioner);

  // compute matrix and right-hand side vector
  hp::FEValues<dim> fe_values_collection(mapping,
                                         fe,
                                         quadrature,
                                         update_gradients | update_values |
                                           update_JxW_values);

  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : system.locally_active_cell_iterators())
    {
      fe_values_collection.reinit(cell->dealii_iterator(),
                                  numbers::invalid_unsigned_int,
                                  numbers::invalid_unsigned_int,
                                  cell->active_fe_index());

      const auto &fe_values = fe_values_collection.get_present_fe_values();

      const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();

      // get indices
      dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      // compute element stiffness matrix
      FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) += fe_values.shape_grad(i, q_index) *
                                   fe_values.shape_grad(j, q_index) *
                                   fe_values.JxW(q_index);
        }

      // compute element vector
      Vector<Number> cell_vector(dofs_per_cell);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          cell_vector(i) +=
            1.0 * fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);

      // assemble
      constraints.distribute_local_to_global(
        cell_matrix, cell_vector, dof_indices, sparse_matrix, rhs);
    }

  sparse_matrix.compress(VectorOperation::values::add);
  rhs.compress(VectorOperation::values::add);

  // choose preconditioner
  TrilinosWrappers::PreconditionAMG preconditioner;
  preconditioner.initialize(sparse_matrix);

  // solve problem
  ReductionControl     solver_control(100, 1.e-10, 1.e-4);
  SolverCG<VectorType> solver(solver_control);
  solver.solve(sparse_matrix, solution, rhs, preconditioner);
  constraints.distribute(solution);
  pcout << solver_control.last_step() << std::endl << std::endl;

  // output result
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  IndexSet           is_all(system.n_dofs());
  is_all.add_range(0, system.n_dofs());
  VectorType solution_root(system.locally_owned_dofs(), is_all, MPI_COMM_WORLD);
  solution_root = solution;
  solution_root.update_ghost_values();

  if (my_rank == 0)
    {
      for (unsigned int i = 0; i < solution_root.size(); ++i)
        pcout << solution_root.begin()[i] << std::endl;
      pcout << std::endl;
    }

  // output result -> Paraview
  GDM::DataOut<dim> data_out(system, mapping, fe_degree_output);
  solution.update_ghost_values();
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  data_out.write_vtu_in_parallel("solution.vtu");
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<1>();
  test<2>();
}
