#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// DGGD headers
#include <gdm/fe_even.h>
#include <gdm/system_dggd.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>


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

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  // Construct the even-degree DGGD system.
  // system_dggd.h creates:
  // 1. a uniform primal grid carrying the nodal coefficients;
  // 2. a dual grid used for DGGD reconstruction and integration.
  GDM::System<dim> system(fe_degree, 1);

  system.subdivided_hyper_cube(n_subdivisions_1D, 0.0, 1.0);
  system.categorize();

  const auto &fe = system.get_fe();

  const auto &vertex_dof_handler = system.get_dof_handler();

  const auto &dggd_triangulation = system.get_triangulation();


  // **************************************************************************
  // Structural checks
  // **************************************************************************

  unsigned int expected_n_dofs     = 1;
  unsigned int expected_n_cells    = 1;
  unsigned int expected_fe_entries = 1;

  for (unsigned int d = 0; d < dim; ++d)
    {
      // N+1 primal nodes in every coordinate direction.
      expected_n_dofs *= n_subdivisions_1D + 1;

      // N+1 dual cells in every coordinate direction.
      expected_n_cells *= n_subdivisions_1D + 1;

      // p+1 position-dependent basis categories in every direction.
      expected_fe_entries *= fe_degree + 1;
    }

  AssertDimension(system.n_dofs(), expected_n_dofs);

  AssertDimension(vertex_dof_handler.n_dofs(), expected_n_dofs);

  AssertDimension(dggd_triangulation.n_active_cells(), expected_n_cells);

  AssertDimension(fe.size(), expected_fe_entries);


  // **************************************************************************
  // Mapping and quadrature
  // **************************************************************************

  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());

  hp::QCollection<dim> projection_quadrature;
  projection_quadrature.push_back(QGauss<dim>(fe_degree + 1));


  // No boundary constraints are imposed in this test.
  AffineConstraints<double> constraints;
  constraints.close();


  // Exact solution:
  // f(x) = x².
  const ScalarFunctionFromFunctionObject<dim> exact_solution(
    [](const Point<dim> &point) { return point[0] * point[0]; });


  // Both vectors contain coefficients associated with the primal nodes.
  VectorType solution_projected;
  VectorType solution_interpolated;

  solution_projected.reinit(system.n_dofs());
  solution_interpolated.reinit(system.n_dofs());


  // **************************************************************************
  // L2 projection
  // **************************************************************************

  if (do_l2_projection)
    {
      // ----------------------------------------------------------------------
      // Create the sparsity pattern
      // ----------------------------------------------------------------------

      TrilinosWrappers::SparsityPattern sparsity_pattern;

      sparsity_pattern.reinit(system.n_dofs(), system.n_dofs());

      // The DGGD system supplies the correct stencil for every dual cell.
      system.create_sparsity_pattern(constraints, sparsity_pattern);

      sparsity_pattern.compress();


      // ----------------------------------------------------------------------
      // Create the mass matrix and right-hand side
      // ----------------------------------------------------------------------

      TrilinosWrappers::SparseMatrix mass_matrix;
      mass_matrix.reinit(sparsity_pattern);

      VectorType right_hand_side;
      right_hand_side.reinit(solution_projected);


      // hp::FEValues is required because different dual cells may use
      // different position-dependent finite elements.
      hp::FEValues<dim> hp_fe_values(mapping,
                                     fe,
                                     projection_quadrature,
                                     update_values | update_JxW_values |
                                       update_quadrature_points);

      std::vector<types::global_dof_index> dof_indices;


      // ----------------------------------------------------------------------
      // Assemble the global mass matrix
      // ----------------------------------------------------------------------

      for (const auto &cell : system.locally_active_cell_iterators())
        {
          const unsigned int active_fe_index = cell->active_fe_index();

          // Reinitialize FEValues on the actual dual-grid cell using the
          // finite-element category assigned by System::categorize().
          hp_fe_values.reinit(cell->dealii_iterator(),
                              numbers::invalid_unsigned_int,
                              numbers::invalid_unsigned_int,
                              active_fe_index);

          const auto &fe_values = hp_fe_values.get_present_fe_values();

          const unsigned int dofs_per_cell =
            fe_values.get_fe().n_dofs_per_cell();

          dof_indices.resize(dofs_per_cell);

          // Obtain the primal-node stencil used by this dual cell.
          cell->get_dof_indices(dof_indices);


          FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

          Vector<double> cell_right_hand_side(dofs_per_cell);


          // Local system:
          // M_ij^K = ∫_K phi_i phi_j dx,
          // b_i^K  = ∫_K f phi_i dx.
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                {
                  cell_right_hand_side(i) +=
                    exact_solution.value(fe_values.quadrature_point(q)) *
                    fe_values.shape_value(i, q) * fe_values.JxW(q);

                  for (const unsigned int j : fe_values.dof_indices())
                    {
                      cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                                fe_values.shape_value(j, q) *
                                                fe_values.JxW(q);
                    }
                }
            }


          constraints.distribute_local_to_global(cell_mass_matrix,
                                                 cell_right_hand_side,
                                                 dof_indices,
                                                 mass_matrix,
                                                 right_hand_side);
        }


      mass_matrix.compress(VectorOperation::values::add);

      right_hand_side.compress(VectorOperation::values::add);


      // Solve
      // M u_projected = b.
      TrilinosWrappers::SolverDirect mass_matrix_solver;

      mass_matrix_solver.initialize(mass_matrix);

      mass_matrix_solver.vmult(solution_projected, right_hand_side);
    }


  // **************************************************************************
  // Interpolation
  // **************************************************************************

  if (do_interpolation)
    {
      // The helper DoFHandler is attached to the uniform primal mesh.
      // Therefore, this produce
      // u_j = f(x_j)
      // at the primal nodes.
      VectorTools::interpolate(mapping,
                               vertex_dof_handler,
                               exact_solution,
                               solution_interpolated);
    }


  // **************************************************************************
  // Linear ParaView output
  // **************************************************************************

  if (do_paraview_lin)
    {
      VectorType analytical_solution;

      analytical_solution.reinit(vertex_dof_handler.n_dofs());

      VectorTools::interpolate(mapping,
                               vertex_dof_handler,
                               exact_solution,
                               analytical_solution);


      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;


      DataOut<dim> data_out;

      data_out.set_flags(flags);

      data_out.attach_dof_handler(vertex_dof_handler);

      data_out.add_data_vector(solution_interpolated,
                               "solution_interpolated_lin");

      data_out.add_data_vector(solution_projected, "solution_projected_lin");

      data_out.add_data_vector(analytical_solution, "solution_analytical");


      data_out.build_patches(
        mapping, 1, DataOut<dim>::CurvedCellRegion::curved_inner_cells);

      data_out.write_vtu_in_parallel("results_lin.vtu",
                                     vertex_dof_handler.get_mpi_communicator());
    }


  // **************************************************************************
  // Reconstructed DGGD ParaView output
  // **************************************************************************

  if (do_paraview)
    {
      DoFHandler<dim> output_dof_handler(dggd_triangulation);

      output_dof_handler.distribute_dofs(FE_DGQ<dim>(fe_degree));


      VectorType analytical_solution;
      VectorType solution_interpolated_output;
      VectorType solution_projected_output;

      analytical_solution.reinit(output_dof_handler.n_dofs());

      solution_interpolated_output.reinit(output_dof_handler.n_dofs());

      solution_projected_output.reinit(output_dof_handler.n_dofs());


      VectorTools::interpolate(mapping,
                               output_dof_handler,
                               exact_solution,
                               analytical_solution);


      // Evaluate the DGGD reconstruction at the support points of FE_DGQ(p).
      const Quadrature<dim> output_points(
        output_dof_handler.get_fe().get_unit_support_points());

      hp::QCollection<dim> output_quadrature;

      output_quadrature.push_back(output_points);


      hp::FEValues<dim> hp_fe_values(mapping,
                                     fe,
                                     output_quadrature,
                                     update_values);


      std::vector<types::global_dof_index> dggd_dof_indices;

      std::vector<types::global_dof_index> output_dof_indices(
        output_dof_handler.get_fe().n_dofs_per_cell());


      for (const auto &cell : system.locally_active_cell_iterators())
        {
          hp_fe_values.reinit(cell->dealii_iterator(),
                              numbers::invalid_unsigned_int,
                              numbers::invalid_unsigned_int,
                              cell->active_fe_index());

          const auto &fe_values = hp_fe_values.get_present_fe_values();


          // Global primal-node stencil used by the DGGD reconstruction.
          dggd_dof_indices.resize(fe_values.get_fe().n_dofs_per_cell());

          cell->get_dof_indices(dggd_dof_indices);


          // Output DoFs belonging to the same geometric dual cell.
          cell->dealii_iterator()
            ->as_dof_handler_iterator(output_dof_handler)
            ->get_dof_indices(output_dof_indices);


          std::vector<double> reconstructed_values(
            fe_values.n_quadrature_points);


          // Reconstruct the interpolated solution.
          fe_values.get_function_values(solution_interpolated,
                                        dggd_dof_indices,
                                        reconstructed_values);

          for (unsigned int i = 0; i < output_dof_indices.size(); ++i)
            {
              solution_interpolated_output[output_dof_indices[i]] =
                reconstructed_values[i];
            }


          // Reconstruct the projected solution.
          fe_values.get_function_values(solution_projected,
                                        dggd_dof_indices,
                                        reconstructed_values);

          for (unsigned int i = 0; i < output_dof_indices.size(); ++i)
            {
              solution_projected_output[output_dof_indices[i]] =
                reconstructed_values[i];
            }
        }


      solution_interpolated_output.compress(VectorOperation::values::insert);

      solution_projected_output.compress(VectorOperation::values::insert);


      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;


      DataOut<dim> data_out;

      data_out.set_flags(flags);

      data_out.attach_dof_handler(output_dof_handler);

      data_out.add_data_vector(analytical_solution, "solution_analytical");

      data_out.add_data_vector(solution_interpolated_output,
                               "solution_interpolated");

      data_out.add_data_vector(solution_projected_output, "solution_projected");


      data_out.build_patches(
        mapping, fe_degree, DataOut<dim>::CurvedCellRegion::curved_inner_cells);

      data_out.write_vtu_in_parallel("results.vtu",
                                     output_dof_handler.get_mpi_communicator());
    }


  // **************************************************************************
  // Compute interpolation and projection errors
  // **************************************************************************

  double error_i = 0.0;
  double error_p = 0.0;


  if (do_compute_error)
    {
      hp::QCollection<dim> error_quadrature;

      error_quadrature.push_back(QGauss<dim>(fe_degree + 3));


      hp::FEValues<dim> hp_fe_values(mapping,
                                     fe,
                                     error_quadrature,
                                     update_values | update_JxW_values |
                                       update_quadrature_points);


      std::vector<types::global_dof_index> dof_indices;


      for (const auto &cell : system.locally_active_cell_iterators())
        {
          hp_fe_values.reinit(cell->dealii_iterator(),
                              numbers::invalid_unsigned_int,
                              numbers::invalid_unsigned_int,
                              cell->active_fe_index());

          const auto &fe_values = hp_fe_values.get_present_fe_values();


          dof_indices.resize(fe_values.get_fe().n_dofs_per_cell());

          cell->get_dof_indices(dof_indices);


          std::vector<double> reconstructed_values(
            fe_values.n_quadrature_points);


          // --------------------------------------------------------------
          // Interpolation error
          // --------------------------------------------------------------

          fe_values.get_function_values(solution_interpolated,
                                        dof_indices,
                                        reconstructed_values);

          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const double difference =
                reconstructed_values[q] -
                exact_solution.value(fe_values.quadrature_point(q));

              error_i += difference * difference * fe_values.JxW(q);
            }


          // --------------------------------------------------------------
          // Projection error
          // --------------------------------------------------------------

          fe_values.get_function_values(solution_projected,
                                        dof_indices,
                                        reconstructed_values);

          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const double difference =
                reconstructed_values[q] -
                exact_solution.value(fe_values.quadrature_point(q));

              error_p += difference * difference * fe_values.JxW(q);
            }
        }


      error_i = std::sqrt(error_i);
      error_p = std::sqrt(error_p);

      const double exact_reproduction_tolerance = 1.0e-11;

      AssertThrow(error_i < exact_reproduction_tolerance,
                  ExcMessage("DGGD interpolation does not reproduce x^2."));

      AssertThrow(error_p < exact_reproduction_tolerance,
                  ExcMessage("DGGD L2 projection does not reproduce x^2."));
    }


  // **************************************************************************
  // Convergence table
  // **************************************************************************

  table.add_value("n_cells", n_subdivisions_1D);

  table.add_value("degree", fe_degree);


  table.add_value("error_i", error_i);

  table.set_scientific("error_i", true);

  table.evaluate_convergence_rates(
    "error_i", "n_cells", ConvergenceTable::RateMode::reduction_rate_log2, 1);


  table.add_value("error_p", error_p);

  table.set_scientific("error_p", true);

  table.evaluate_convergence_rates(
    "error_p", "n_cells", ConvergenceTable::RateMode::reduction_rate_log2, 1);
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);


  const int dim = 2;

  // DGGD uses an even polynomial degree.
  const unsigned int fe_degree = 2;


  const unsigned int n_subdivisions_1D_min = 10;

  const unsigned int n_subdivisions_1D_max = 10;

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
