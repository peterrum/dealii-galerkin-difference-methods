// Invert mass matrix.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>

#include <gdm/data_out.h>
#include <gdm/system.h>
#include <gdm/vector_tools.h>

#include <fstream>



template <int dim>
void
test()
{
  const unsigned int n_subdivisions = 40;
  const unsigned int fe_degree      = 3;
  const unsigned int n_components   = 2;

  using Number     = double;
  using VectorType = Vector<Number>;

  FunctionFromFunctionObjects<dim> exact_solution(
    [&](const auto &p, const auto c) {
      const double a = numbers::PI;
      const double x = p[0];
      const double y = p[1];

      if (c == 0)
        return std::sin(a * x) * std::sin(a * x) * std::cos(a * y) *
               std::sin(a * y);
      else if (c == 1)
        return -std::cos(a * x) * std::sin(a * x) * std::sin(a * y) *
               std::sin(a * y);

      AssertThrow(false, ExcNotImplemented());

      return 0.0;
    },
    dim);

  FunctionFromFunctionObjects<dim> function(
    [&](const auto &p, const auto c) {
      const double a = numbers::PI;
      const double x = p[0];
      const double y = p[1];

      if (c == 0)
        return 6 * a * a * std::sin(a * x) * std::sin(a * x) * std::sin(a * y) *
                 std::cos(a * y) -
               2 * a * a * std::sin(a * y) * std::cos(a * x) * std::cos(a * x) *
                 std::cos(a * y);
      else if (c == 1)
        return -6 * a * a * std::sin(a * x) * std::sin(a * y) *
                 std::sin(a * y) * std::cos(a * x) +
               2 * a * a * std::sin(a * x) * std::cos(a * x) * std::cos(a * y) *
                 std::cos(a * y);

      AssertThrow(false, ExcNotImplemented());

      return 0.0;
    },
    n_components);

  // Create GDM system
  GDM::System<dim> system(fe_degree, n_components);

  // Create mesh
  system.subdivided_hyper_cube(n_subdivisions);

  // Create finite elements
  const auto &fe = system.get_fe();

  // Create mapping
  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());

  // Create quadrature
  hp::QCollection<dim> quadrature;
  quadrature.push_back(QGauss<dim>(fe_degree + 1));

  // Create constraints
  AffineConstraints<Number> constraints;
  system.make_zero_boundary_constraints(constraints);
  constraints.close();

  // Categorize cells
  system.categorize();

  // Create sparsity pattern and allocate sparse matrix
  const unsigned int n_dofs = system.n_dofs();

  DynamicSparsityPattern dsp(n_dofs);
  system.create_sparsity_pattern(constraints, dsp);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<Number> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  // create vectors
  VectorType rhs(n_dofs);
  VectorType solution(n_dofs);

  // compute matrix and right-hand side vector
  hp::FEValues<dim> fe_values_collection(mapping,
                                         fe,
                                         quadrature,
                                         update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points);

  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : system.locally_active_cell_iterators())
    {
      fe_values_collection.reinit(cell->dealii_iterator(),
                                  numbers::invalid_unsigned_int,
                                  numbers::invalid_unsigned_int,
                                  cell->active_fe_index());

      const auto &fe_values = fe_values_collection.get_present_fe_values();

      FEValuesViews::Vector<dim> velocities(fe_values, 0);

      const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();

      // get indices
      dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      // compute element stiffness matrix
      FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const auto JxW = fe_values.JxW(q_index);

          for (const unsigned int i : fe_values.dof_indices())
            {
              const auto eps_v_i = velocities.symmetric_gradient(i, q_index);
              for (const unsigned int j : fe_values.dof_indices())
                {
                  const auto eps_u_j =
                    velocities.symmetric_gradient(j, q_index);
                  cell_matrix(i, j) +=
                    2.0 * scalar_product(eps_u_j, eps_v_i) * JxW;
                }
            }
        }

      // compute element vector
      Vector<Number> cell_vector(dofs_per_cell);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          cell_vector(i) +=
            function.value(
              fe_values.quadrature_point(q_index),
              fe_values.get_fe().system_to_component_index(i).first) *
            fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);

      // assemble
      constraints.distribute_local_to_global(
        cell_matrix, cell_vector, dof_indices, sparse_matrix, rhs);
    }

  // choose preconditioner
  PreconditionJacobi<SparseMatrix<Number>> preconditioner;
  preconditioner.initialize(sparse_matrix);

  // solve problem
  ReductionControl     solver_control(100, 1.e-10, 1.e-8);
  SolverCG<VectorType> solver(solver_control);
  solver.solve(sparse_matrix, solution, rhs, preconditioner);

  // computer error
  Vector<Number> cell_wise_error;
  GDM::VectorTools::integrate_difference(mapping,
                                         system,
                                         solution,
                                         exact_solution,
                                         cell_wise_error,
                                         quadrature,
                                         VectorTools::NormType::L2_norm);
  const auto error =
    VectorTools::compute_global_error(system.get_triangulation(),
                                      cell_wise_error,
                                      VectorTools::NormType::L2_norm);

  std::cout << "error: " << error << std::endl;
}


int
main()
{
  test<2>();
}
