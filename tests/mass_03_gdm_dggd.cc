#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>

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

#include <gdm/fe_even.h>


using namespace dealii;


template <int dim>
void
test(const unsigned int n_subdivisions_1D,
     const unsigned int fe_degree,
     ConvergenceTable  &table)
{
  // **************************************************************************
  // setup
  // **************************************************************************

  const bool do_l2_projection = true;
  const bool do_interpolation = true;
  const bool do_paraview_lin  = true;
  const bool do_paraview      = true;
  const bool do_compute_error = true;
  const bool even_polynomial  = fe_degree % 2 == 0;

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());
  QGauss<dim> quadrature(fe_degree + 1);

  AffineConstraints<double> constraints;
  constraints.close();

  Triangulation<dim> vertex_tria;
  GridGenerator::subdivided_hyper_cube(vertex_tria,
                                       n_subdivisions_1D,
                                       0.0,
                                       +1.0);

  Triangulation<dim> dual_tria;

  if (even_polynomial)
    {
      // DGGD dual mesh: N+1 cells, first/last half-width
      const double h = 1.0 / n_subdivisions_1D; // domain is [0,1]

      std::vector<std::vector<double>> step_sizes(dim);
      for (unsigned int d = 0; d < dim; ++d)
        {
          step_sizes[d].push_back(h / 2.0); //  <- first cell
          for (unsigned int i = 1; i < n_subdivisions_1D; ++i)
            step_sizes[d].push_back(h);     //  <- middle cells
          step_sizes[d].push_back(h / 2.0); //  <- last cell
        }

      Point<dim> p1, p2;
      for (unsigned int d = 0; d < dim; ++d)
        {
          p1[d] = 0.0;
          p2[d] = 1.0;
        }

      GridGenerator::subdivided_hyper_rectangle(
        dual_tria, step_sizes, p1, p2, true);
    }

  // select triangulation for cell loops
  const auto &comp_tria = even_polynomial ? dual_tria : vertex_tria;

  DoFHandler<dim> vertex_dof_handler(vertex_tria);
  vertex_dof_handler.distribute_dofs(FE_Q<dim>(1));

  const auto hp_fe =
    GDM::generate_fe_collection<dim>(GDM::generate_polynomials_1D(fe_degree),
                                     1);

  const auto        &fe              = hp_fe;
  const unsigned int n_dofs_per_cell = fe.max_dofs_per_cell();

  const auto exact_solution =
    std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
      [](const auto &p) { return std::sin(p[0]); });

  // cells per direction: N+1 for DGGD, N for continuous
  std::array<unsigned int, dim> cell_shape;
  for (unsigned int d = 0; d < dim; ++d)
    cell_shape[d] =
      even_polynomial ? (n_subdivisions_1D + 1) : n_subdivisions_1D;

  // Category per direction: fe_degree+1 for DGGD, fe_degree for continuous
  const unsigned int category_count =
    even_polynomial ? fe_degree + 1 : fe_degree;

  const auto get_active_fe_index = [&](const auto &cell) {
    //  flat cell number -> (row, col, ...)
    const auto indices =
      GDM::index_to_indices<dim>(cell->active_cell_index(), cell_shape);

    // same formula as before, applied per direction
    std::array<unsigned int, dim> category_indices;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const unsigned int index = indices[d];
        category_indices[d]      = (index < (fe_degree / 2) ?
                                      index :
                                      (index < (n_subdivisions_1D - fe_degree / 2) ?
                                         (fe_degree / 2) :
                                         (fe_degree + index - n_subdivisions_1D)));
      }

    // recombine into one flat FE-collection index
    return GDM::indices_to_index<dim>(category_indices, category_count);
  };

  const auto get_dof_indices = [&](const auto &cell) {
    const auto active_fe_index = get_active_fe_index(cell);
    const auto n_dofs_per_cell = hp_fe[active_fe_index].n_dofs_per_cell();

    //  flat cell number -> (row, col, ...)
    const auto indices =
      GDM::index_to_indices<dim>(cell->active_cell_index(), cell_shape);

    // window start, per direction (centered-on-1-node vs straddle-2-nodes)
    std::array<unsigned int, dim> offset_reference;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const unsigned int index = indices[d];
        if (even_polynomial)
          offset_reference[d] =
            (index < fe_degree / 2) ?
              0 :
              std::min(n_subdivisions_1D - fe_degree, index - fe_degree / 2);
        else
          offset_reference[d] =
            (index < fe_degree / 2) ?
              0 :
              (std::min(n_subdivisions_1D, index + fe_degree / 2 + 1) -
               fe_degree);
      }

    // global node-grid shape: always N+1 per direction
    std::array<unsigned int, dim> n_dofs;
    for (unsigned int d = 0; d < dim; ++d)
      n_dofs[d] = n_subdivisions_1D + 1;

    // walk the (fe_degree+1)^dim window, flatten each point
    std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
    for (unsigned int k = 0, c = 0; k <= ((dim >= 3) ? fe_degree : 0); ++k)
      for (unsigned int j = 0; j <= ((dim >= 2) ? fe_degree : 0); ++j)
        for (unsigned int i = 0; i <= fe_degree; ++i, ++c)
          {
            auto offset = offset_reference;
            if (dim >= 1)
              offset[0] += i;
            if (dim >= 2)
              offset[1] += j;
            if (dim >= 3)
              offset[2] += k;

            dof_indices[c] = GDM::indices_to_index<dim>(offset, n_dofs);
          }

    return dof_indices;
  };

  VectorType solution_projected, solution_interpolated;
  solution_projected.reinit(vertex_dof_handler.n_dofs());
  solution_interpolated.reinit(vertex_dof_handler.n_dofs());

  // **************************************************************************
  // perform l2 projection
  // **************************************************************************

  if (do_l2_projection)
    {
      TrilinosWrappers::SparsityPattern sparsity_pattern;
      TrilinosWrappers::SparseMatrix    mass_matrix;

      VectorType rhs;
      rhs.reinit(solution_projected);

      {
        sparsity_pattern.reinit(vertex_dof_handler.n_dofs(),
                                vertex_dof_handler.n_dofs());

        for (const auto &cell : comp_tria.active_cell_iterators())
          {
            const auto dof_indices = get_dof_indices(cell);

            constraints.add_entries_local_to_global(dof_indices,
                                                    sparsity_pattern);
          }

        sparsity_pattern.compress();
      }

      mass_matrix.reinit(sparsity_pattern);

      hp::FEValues<dim> hp_fe_values(mapping,
                                     fe,
                                     hp::QCollection<dim>(quadrature),
                                     update_JxW_values | update_values |
                                       update_quadrature_points);

      for (const auto &cell : comp_tria.active_cell_iterators())
        {
          const unsigned int active_fe_index = get_active_fe_index(cell);
          hp_fe_values.reinit(cell,
                              numbers::invalid_unsigned_int,
                              numbers::invalid_unsigned_int,
                              active_fe_index);

          const auto &fe_values = hp_fe_values.get_present_fe_values();

          FullMatrix<double> mass_cell_matrix(n_dofs_per_cell, n_dofs_per_cell);

          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            for (const unsigned int i : fe_values.dof_indices())
              for (const unsigned int j : fe_values.dof_indices())
                mass_cell_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                          fe_values.shape_value(j, q_index) *
                                          fe_values.JxW(q_index);

          Vector<double> cell_vector(n_dofs_per_cell);
          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            for (const unsigned int i : fe_values.dof_indices())
              cell_vector(i) +=
                exact_solution->value(fe_values.quadrature_point(q_index)) *
                fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);

          const auto dof_indices = get_dof_indices(cell);

          constraints.distribute_local_to_global(
            mass_cell_matrix, cell_vector, dof_indices, mass_matrix, rhs);
        }

      mass_matrix.compress(VectorOperation::values::add);
      rhs.compress(VectorOperation::values::add);

      TrilinosWrappers::SolverDirect mass_matrix_solver;
      mass_matrix_solver.initialize(mass_matrix);

      mass_matrix_solver.vmult(solution_projected, rhs);
    }


  // **************************************************************************
  // interpolate
  // **************************************************************************

  if (do_interpolation)
    {
      const double                  h = 1.0 / n_subdivisions_1D;
      std::array<unsigned int, dim> node_shape;
      for (unsigned int d = 0; d < dim; ++d)
        node_shape[d] = n_subdivisions_1D + 1;

      for (unsigned int index = 0; index < vertex_dof_handler.n_dofs(); ++index)
        {
          const auto indices = GDM::index_to_indices<dim>(index, node_shape);

          Point<dim> p;
          for (unsigned int d = 0; d < dim; ++d)
            p[d] = indices[d] * h;

          solution_interpolated[index] = exact_solution->value(p);
        }
    }



  // **************************************************************************
  // postprocess: write paraview results
  // **************************************************************************

  if (do_paraview_lin)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_triangulation(vertex_tria);
      data_out.add_data_vector(vertex_dof_handler,
                               solution_interpolated,
                               "solution_interpolated_lin");
      data_out.add_data_vector(vertex_dof_handler,
                               solution_projected,
                               "solution_projected_lin");

      DoFHandler<dim> dof_handler_solution;

      if (exact_solution)
        {
          dof_handler_solution.reinit(vertex_tria);
          dof_handler_solution.distribute_dofs(FE_Q<dim>(1));
          VectorType analytical_solution;

          analytical_solution.reinit(dof_handler_solution.n_dofs());
          VectorTools::interpolate(mapping,
                                   dof_handler_solution,
                                   *exact_solution,
                                   analytical_solution);
          data_out.add_data_vector(dof_handler_solution,
                                   analytical_solution,
                                   "solution_analytical");
        }

      data_out.build_patches(
        mapping, 1, DataOut<dim>::CurvedCellRegion::curved_inner_cells);
      data_out.write_vtu_in_parallel("results_lin.vtu",
                                     vertex_dof_handler.get_mpi_communicator());
    }

  if (do_paraview)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      DoFHandler<dim> dof_handler_solution;

      dof_handler_solution.reinit(comp_tria);
      dof_handler_solution.distribute_dofs(FE_Q<dim>(fe_degree));

      VectorType analytical_solution, solution_interpolated_fe,
        solution_projected_fe;

      solution_interpolated_fe.reinit(dof_handler_solution.n_dofs());
      solution_projected_fe.reinit(dof_handler_solution.n_dofs());

      if (exact_solution)
        {
          analytical_solution.reinit(dof_handler_solution.n_dofs());
          VectorTools::interpolate(mapping,
                                   dof_handler_solution,
                                   *exact_solution,
                                   analytical_solution);
          data_out.add_data_vector(dof_handler_solution,
                                   analytical_solution,
                                   "solution_analytical");
        }

      hp::FEValues<dim> hp_fe_values(
        mapping,
        fe,
        hp::QCollection<dim>(Quadrature<dim>(
          dof_handler_solution.get_fe().get_unit_support_points())),
        update_values);

      for (const auto &cell : comp_tria.active_cell_iterators())
        {
          const unsigned int active_fe_index = get_active_fe_index(cell);
          hp_fe_values.reinit(cell,
                              numbers::invalid_unsigned_int,
                              numbers::invalid_unsigned_int,
                              active_fe_index);

          const auto &fe_values = hp_fe_values.get_present_fe_values();

          const auto dof_indices = get_dof_indices(cell);

          std::vector<types::global_dof_index> dof_indices_fe(
            dof_handler_solution.get_fe().n_dofs_per_cell());

          cell->as_dof_handler_iterator(dof_handler_solution)
            ->get_dof_indices(dof_indices_fe);

          std::vector<double> quadrature_values(fe_values.n_quadrature_points);

          // interpolated solution
          fe_values.get_function_values(solution_interpolated,
                                        dof_indices,
                                        quadrature_values);

          for (unsigned int i = 0; i < dof_indices_fe.size(); ++i)
            solution_interpolated_fe[dof_indices_fe[i]] = quadrature_values[i];


          // projected solution
          fe_values.get_function_values(solution_projected,
                                        dof_indices,
                                        quadrature_values);

          for (unsigned int i = 0; i < dof_indices_fe.size(); ++i)
            solution_projected_fe[dof_indices_fe[i]] = quadrature_values[i];
        }

      data_out.add_data_vector(dof_handler_solution,
                               solution_interpolated_fe,
                               "solution_interpolated");
      data_out.add_data_vector(dof_handler_solution,
                               solution_projected_fe,
                               "solution_projected");

      // write data
      data_out.build_patches(
        mapping, fe_degree, DataOut<dim>::CurvedCellRegion::curved_inner_cells);
      data_out.write_vtu_in_parallel("results.vtu",
                                     vertex_dof_handler.get_mpi_communicator());
    }



  // **************************************************************************
  // postprocess: compute error
  // **************************************************************************

  double error_i = 0.0;
  double error_p = 0.0;
  if (do_compute_error && exact_solution)
    {
      hp::FEValues<dim> hp_fe_values(
        mapping,
        fe,
        hp::QCollection<dim>(QGauss<dim>(fe_degree + 3)),
        update_values | update_JxW_values | update_quadrature_points);

      error_i = 0.0;

      for (const auto &cell : comp_tria.active_cell_iterators())
        {
          const unsigned int active_fe_index = get_active_fe_index(cell);
          hp_fe_values.reinit(cell,
                              numbers::invalid_unsigned_int,
                              numbers::invalid_unsigned_int,
                              active_fe_index);

          const auto &fe_values = hp_fe_values.get_present_fe_values();

          const auto dof_indices = get_dof_indices(cell);

          std::vector<double> quadrature_values(fe_values.n_quadrature_points);

          // interpolated solution
          fe_values.get_function_values(solution_interpolated,
                                        dof_indices,
                                        quadrature_values);

          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            error_i += std::pow(quadrature_values[q_index] -
                                  exact_solution->value(
                                    fe_values.quadrature_point(q_index)),
                                2.0) *
                       fe_values.JxW(q_index);

          // projected solution
          fe_values.get_function_values(solution_projected,
                                        dof_indices,
                                        quadrature_values);

          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            error_p += std::pow(quadrature_values[q_index] -
                                  exact_solution->value(
                                    fe_values.quadrature_point(q_index)),
                                2.0) *
                       fe_values.JxW(q_index);
        }

      error_i = std::sqrt(error_i);
      error_p = std::sqrt(error_p);
    }


  // **************************************************************************
  // postprocess: create convergence table
  // **************************************************************************

  table.add_value("n_cells", n_subdivisions_1D);
  table.add_value("degree", fe_degree);

  table.add_value("error_i", error_i);
  table.set_scientific("error_i", true);
  table.evaluate_convergence_rates(
    "error_i", "n_cells", ConvergenceTable::RateMode::reduction_rate_log2, dim);

  table.add_value("error_p", error_p);
  table.set_scientific("error_p", true);
  table.evaluate_convergence_rates(
    "error_p", "n_cells", ConvergenceTable::RateMode::reduction_rate_log2, dim);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const int          dim       = 2;
  const unsigned int fe_degree = 4;

  const unsigned int n_subdivisions_1D_min  = 5;
  const unsigned int n_subdivisions_1D_max  = 40;
  const unsigned int n_subdivisions_1D_step = 5;
  ConvergenceTable   table;

  for (unsigned int n_subdivisions_1D = n_subdivisions_1D_min;
       n_subdivisions_1D <=
       std::max(n_subdivisions_1D_min, n_subdivisions_1D_max);
       n_subdivisions_1D += n_subdivisions_1D_step)
    {
      test<dim>(n_subdivisions_1D, fe_degree, table);
    }

  table.write_text(std::cout);
}
