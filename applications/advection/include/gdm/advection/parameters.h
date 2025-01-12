#pragma once

using namespace dealii;

template <unsigned int dim>
struct Parameters
{
  // general settings
  unsigned int fe_degree;

  // geometry
  unsigned int n_subdivisions_1D;

  // mass matrix

  // stiffness matrix

  // time stepping
  std::shared_ptr<Function<dim>> exact_solution;
  std::shared_ptr<Function<dim>> exact_solution_der;
  double                         start_t;
  double                         end_t;
  double                         cfl;

  // linear solver

  // advection
  std::shared_ptr<Function<dim>> advection;

  // level set field
  std::shared_ptr<Function<dim>> level_set_function;

  // output
};
