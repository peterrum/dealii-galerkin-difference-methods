#include <deal.II/base/convergence_table.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/time_stepping.templates.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <gdm/fe.h>


using namespace dealii;


template <int dim>
void
test(const unsigned int n_subdivisions_1D,
     const unsigned int fe_degree,
     ConvergenceTable  &table)
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());
  QGauss<dim> quadrature(fe_degree + 1);

  AffineConstraints<double> constraints;
  constraints.close();

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions_1D, 0.0, +1.0);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(1));

  const auto hp_fe =
    GDM::generate_fe_collection<dim>(GDM::generate_polynomials_1D(fe_degree),
                                     1);

  const auto        &fe              = hp_fe;
  const unsigned int n_dofs_per_cell = fe.max_dofs_per_cell();

  const auto exact_solution =
    std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
      [](const auto &p) { return p[0]; });

  const auto get_active_fe_index = [&](const auto &cell) {
    const unsigned int index = cell->active_cell_index();
    return (index < (fe_degree / 2) ?
              index :
              (index < (n_subdivisions_1D - fe_degree / 2) ?
                 (fe_degree / 2) :
                 (fe_degree + index - n_subdivisions_1D)));
  };

  VectorType solution;
  solution.reinit(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler, *exact_solution, solution);

  TrilinosWrappers::SparsityPattern sparsity_pattern;
  TrilinosWrappers::SparseMatrix    mass_matrix;

  // allocate memory for mass matrix
  {
    sparsity_pattern.reinit(dof_handler.n_dofs(), dof_handler.n_dofs());

    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : tria.active_cell_iterators())
      if (true)
        {
          dof_indices.resize(n_dofs_per_cell);

          const unsigned int active_fe_index = get_active_fe_index(cell);

          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            dof_indices[i] = cell->active_cell_index() - active_fe_index + i;

          constraints.add_entries_local_to_global(dof_indices,
                                                  sparsity_pattern);
        }

    sparsity_pattern.compress();

    mass_matrix.reinit(sparsity_pattern);
  }

  // compute mass and stiffness matrix
  hp::FEValues<dim> hp_fe_values(mapping,
                                 fe,
                                 hp::QCollection<dim>(quadrature),
                                 update_JxW_values | update_values |
                                   update_gradients);


  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : tria.active_cell_iterators())
    if (true)
      {
        const unsigned int active_fe_index = get_active_fe_index(cell);
        hp_fe_values.reinit(cell,
                            numbers::invalid_unsigned_int,
                            numbers::invalid_unsigned_int,
                            active_fe_index);

        const auto &fe_values = hp_fe_values.get_present_fe_values();

        FullMatrix<double> mass_cell_matrix(n_dofs_per_cell, n_dofs_per_cell);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              mass_cell_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                        fe_values.shape_value(j, q_index) *
                                        fe_values.JxW(q_index);

        // get indices
        dof_indices.resize(n_dofs_per_cell);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          dof_indices[i] = cell->active_cell_index() - active_fe_index + i;

        constraints.distribute_local_to_global(mass_cell_matrix,
                                               dof_indices,
                                               mass_matrix);
      }

  mass_matrix.compress(VectorOperation::values::add);

  // create direct solver
  TrilinosWrappers::SolverDirect mass_matrix_solver;
  mass_matrix_solver.initialize(mass_matrix);

  unsigned int counter = 0;
  double       error   = 0.0;

  const auto postprocess = [&](const double time, const VectorType &solution) {
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
            exact_solution->set_time(time);

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

            hp::FEValues<dim> hp_fe_values(
              mapping,
              fe,
              hp::QCollection<dim>(Quadrature<dim>(
                dof_handler_solution.get_fe().get_unit_support_points())),
              update_values);

            for (const auto &cell : tria.active_cell_iterators())
              if (true)
                {
                  const unsigned int active_fe_index =
                    get_active_fe_index(cell);
                  hp_fe_values.reinit(cell,
                                      numbers::invalid_unsigned_int,
                                      numbers::invalid_unsigned_int,
                                      active_fe_index);

                  const auto &fe_values = hp_fe_values.get_present_fe_values();

                  std::vector<types::global_dof_index> dof_indices(
                    fe_values.dofs_per_cell);

                  for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
                    dof_indices[i] =
                      cell->active_cell_index() - active_fe_index + i;

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

        const std::string file_name =
          "results_" + std::to_string(counter) + ".vtu";
        counter++;

        // write data
        data_out.build_patches(
          mapping,
          fe_degree,
          DataOut<dim>::CurvedCellRegion::curved_inner_cells);
        data_out.write_vtu_in_parallel(file_name,
                                       dof_handler.get_mpi_communicator());
      }


    if (exact_solution)
      {
        exact_solution->set_time(time);

        hp::FEValues<dim> hp_fe_values(
          mapping,
          fe,
          hp::QCollection<dim>(QGauss<dim>(fe_degree + 3)),
          update_values | update_JxW_values | update_quadrature_points);

        error = 0.0;

        for (const auto &cell : tria.active_cell_iterators())
          if (true)
            {
              const unsigned int active_fe_index = get_active_fe_index(cell);
              hp_fe_values.reinit(cell,
                                  numbers::invalid_unsigned_int,
                                  numbers::invalid_unsigned_int,
                                  active_fe_index);

              const auto &fe_values = hp_fe_values.get_present_fe_values();

              std::vector<types::global_dof_index> dof_indices(
                fe_values.dofs_per_cell);

              for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
                dof_indices[i] =
                  cell->active_cell_index() - active_fe_index + i;

              std::vector<double> quadrature_values(
                fe_values.n_quadrature_points);
              fe_values.get_function_values(solution,
                                            dof_indices,
                                            quadrature_values);

              for (const unsigned int q_index :
                   fe_values.quadrature_point_indices())
                error += std::pow(quadrature_values[q_index] -
                                    exact_solution->value(
                                      fe_values.quadrature_point(q_index)),
                                  2.0) *
                         fe_values.JxW(q_index);
            }

        error = std::sqrt(error);
      }
  };


  postprocess(0.0, solution);

  table.add_value("n_cells", n_subdivisions_1D);
  table.add_value("degree", fe_degree);

  table.add_value("error", error);
  table.set_scientific("error", true);
  table.evaluate_convergence_rates(
    "error", "n_cells", ConvergenceTable::RateMode::reduction_rate_log2, dim);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const int          dim                    = 1;
  const unsigned int fe_degree              = 3;
  const unsigned int n_subdivisions_1D_min  = 10;
  const unsigned int n_subdivisions_1D_max  = 40;
  const unsigned int n_subdivisions_1D_step = 10;

  ConvergenceTable table;

  for (unsigned int n_subdivisions_1D = n_subdivisions_1D_min;
       n_subdivisions_1D <=
       std::max(n_subdivisions_1D_min, n_subdivisions_1D_max);
       n_subdivisions_1D += n_subdivisions_1D_step)
    {
      test<dim>(n_subdivisions_1D, fe_degree, table);
    }

  table.write_text(std::cout);
}