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
  const unsigned int n_components   = 1;

  using Number     = double;
  using VectorType = Vector<Number>;

  FunctionFromFunctionObjects<dim> function(
    [&](const auto &p, const auto c) { return p[0] + c; }, n_components);

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
                                         update_values | update_JxW_values |
                                           update_quadrature_points);

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
              cell_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                   fe_values.shape_value(j, q_index) *
                                   fe_values.JxW(q_index);
        }

      // compute element vector
      Vector<Number> cell_vector(dofs_per_cell);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          cell_vector(i) +=
            function.value(fe_values.quadrature_point(q_index)) *
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
                                         function,
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
