#pragma once

using namespace dealii;

template <unsigned int dim>
struct Parameters
{
  // general settings
  unsigned int fe_degree;
  unsigned int n_components;

  // geometry
  unsigned int n_subdivisions_1D;
  double       geometry_left;
  double       geometry_right;

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
  unsigned int                   level_set_fe_degree;
  std::shared_ptr<Function<dim>> level_set_function;

  // output
};
