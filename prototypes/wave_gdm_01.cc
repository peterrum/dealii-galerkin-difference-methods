#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <gdm/fe.h>


using namespace dealii;

template <int dim>
void
test()
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  const unsigned int fe_degree         = 3;
  const unsigned int n_subdivisions_1D = 20;
  const unsigned int n_ghost_cells     = fe_degree / 2;
  const double       dx                = 2.0 / n_subdivisions_1D;

  MappingQ1<dim> mapping;
  QGauss<dim>    quadrature(fe_degree + 1);

  AffineConstraints<double> constraints;
  constraints.close();

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria,
                                       n_subdivisions_1D + 2 * n_ghost_cells,
                                       -1.0 - dx * n_ghost_cells,
                                       +1.0 + dx * n_ghost_cells);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(1));

  const auto hp_fe =
    GDM::generate_fe_collection<dim>(GDM::generate_polynomials_1D(fe_degree),
                                     1);

  const auto &fe = hp_fe[n_ghost_cells];

  const auto exact_solution =
    std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
      [](const auto t, const auto &p) {
        const auto r = p.norm();

        if (dim == 1)
          {
            const auto wave_number = 1.5 * numbers::PI;
            return std::cos(wave_number * r) * std::cos(wave_number * t);
          }
        else
          AssertThrow(false, ExcNotImplemented());
      });

  VectorType solution;
  solution.reinit(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, *exact_solution, solution);

  TrilinosWrappers::SparsityPattern sparsity_pattern;
  TrilinosWrappers::SparseMatrix    sparse_matrix;

  // allocate memory for mass matrix
  {
    sparsity_pattern.reinit(dof_handler.n_dofs(), dof_handler.n_dofs());

    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : tria.active_cell_iterators())
      if ((cell->active_cell_index() >= n_ghost_cells) &&
          (cell->active_cell_index() < n_subdivisions_1D + n_ghost_cells))
        {
          dof_indices.resize(fe.n_dofs_per_cell());

          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            dof_indices[i] = cell->active_cell_index() - n_ghost_cells + i;

          constraints.add_entries_local_to_global(dof_indices,
                                                  sparsity_pattern);
        }

    sparsity_pattern.compress();

    sparse_matrix.reinit(sparsity_pattern);
  }

  // compute mass matrix


  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_JxW_values | update_values);


  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : tria.active_cell_iterators())
    if ((cell->active_cell_index() >= n_ghost_cells) &&
        (cell->active_cell_index() < n_subdivisions_1D + n_ghost_cells))
      {
        fe_values.reinit(cell);

        FullMatrix<double> cell_matrix(fe.n_dofs_per_cell(),
                                       fe.n_dofs_per_cell());

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                   fe_values.shape_value(j, q_index) *
                                   fe_values.JxW(q_index);

        // get indices
        dof_indices.resize(fe.n_dofs_per_cell());

        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          dof_indices[i] = cell->active_cell_index() - n_ghost_cells + i;

        constraints.distribute_local_to_global(cell_matrix,
                                               dof_indices,
                                               sparse_matrix);
      }

  sparse_matrix.compress(VectorOperation::values::add);

  // create direct solver
  TrilinosWrappers::SolverDirect solver_direct;
  solver_direct.initialize(sparse_matrix);

  if (true)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_triangulation(tria);
      data_out.add_data_vector(dof_handler, solution, "solution_lin");


      DoFHandler<dim> dof_handler_solution;
      if (exact_solution)
        {
          dof_handler_solution.reinit(tria);
          dof_handler_solution.distribute_dofs(FE_Q<dim>(fe_degree));

          LinearAlgebra::distributed::Vector<double> analytical_solution(
            dof_handler_solution.n_dofs());
          VectorTools::interpolate(mapping,
                                   dof_handler_solution,
                                   *exact_solution,
                                   analytical_solution);
          data_out.add_data_vector(dof_handler_solution,
                                   analytical_solution,
                                   "solution_ana");

          analytical_solution = 0.0;

          FEValues<dim> fe_values(
            mapping,
            fe,
            dof_handler_solution.get_fe().get_unit_support_points(),
            update_values);

          for (const auto &cell : tria.active_cell_iterators())
            if ((cell->active_cell_index() >= n_ghost_cells) &&
                (cell->active_cell_index() < n_subdivisions_1D + n_ghost_cells))
              {
                fe_values.reinit(cell);

                std::vector<types::global_dof_index> dof_indices(
                  fe_values.dofs_per_cell);

                for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
                  dof_indices[i] =
                    cell->active_cell_index() - n_ghost_cells + i;

                std::vector<double> quadrature_values(
                  fe_values.n_quadrature_points);
                fe_values.get_function_values(solution,
                                              dof_indices,
                                              quadrature_values);

                cell->as_dof_handler_iterator(dof_handler_solution)
                  ->get_dof_indices(dof_indices);

                for (unsigned int i = 0; i < dof_indices.size(); ++i)
                  analytical_solution[dof_indices[i]] = quadrature_values[i];
              }

          data_out.add_data_vector(dof_handler_solution,
                                   analytical_solution,
                                   "solution");
        }

      const std::string file_name = "results.vtu";

      // write data
      data_out.build_patches(
        mapping, fe_degree, DataOut<dim>::CurvedCellRegion::curved_inner_cells);
      data_out.write_vtu_in_parallel(file_name,
                                     dof_handler.get_mpi_communicator());
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const int dim = 1;

  test<dim>();
}