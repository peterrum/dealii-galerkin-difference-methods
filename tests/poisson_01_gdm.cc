// Test Poisson problem: serial, 1D.

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

template <int dim, typename Number = double>
class RightHandSideFunction : public dealii::Function<dim, Number>
{
public:
  RightHandSideFunction()
  {
    AssertDimension(dim, 1);
  }

  virtual double
  value(const dealii::Point<dim> &, const unsigned int = 1) const override
  {
    return 1.0;
  }

private:
};

template <int dim, typename Number = double>
class ExactSolution : public dealii::Function<dim, Number>
{
public:
  ExactSolution()
  {
    AssertDimension(dim, 1);
  }

  virtual double
  value(const dealii::Point<dim> &p, const unsigned int = 1) const override
  {
    return 0.125 - 0.5 * (p[0] - 0.5) * (p[0] - 0.5);
  }

private:
};



template <int dim>
void
test(const unsigned int fe_degree)
{
  const unsigned int n_subdivisions   = 10;
  const unsigned int n_components     = 1;
  const unsigned int fe_degree_output = 2;

  using Number     = double;
  using VectorType = Vector<Number>;

  RightHandSideFunction<dim> right_hand_side_function;
  ExactSolution<dim>         exact_solution;

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
                                         update_gradients | update_values |
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
          cell_vector(i) += right_hand_side_function.value(
                              fe_values.quadrature_point(q_index)) *
                            fe_values.shape_value(i, q_index) *
                            fe_values.JxW(q_index);

      // assemble
      constraints.distribute_local_to_global(
        cell_matrix, cell_vector, dof_indices, sparse_matrix, rhs);
    }

  // choose preconditioner
  PreconditionIdentity preconditioner;

  // solve problem
  ReductionControl     solver_control(100, 1.e-10, 1.e-4);
  SolverCG<VectorType> solver(solver_control);
  solver.solve(sparse_matrix, solution, rhs, preconditioner);
  std::cout << solver_control.last_step() << std::endl << std::endl;

  // output result
  for (const auto &value : solution)
    std::cout << value << std::endl;



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

  printf("%8.5f %14.8f\n\n", 0.0, error);

  // output result -> Paraview
  GDM::DataOut<dim> data_out(system, mapping, fe_degree_output);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream file("solution.vtu");
  data_out.write_vtu(file);
}


int
main()
{
  for (const unsigned int fe_degree : {1, 3, 5, 7, 9})
    test<1>(fe_degree);
}
