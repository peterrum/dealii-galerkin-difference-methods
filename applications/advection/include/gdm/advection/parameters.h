#pragma once

using namespace dealii;

template <unsigned int dim>
struct Parameters
{
  // general settings
  unsigned int fe_degree;
  unsigned int n_components;
  bool         composite = false;

  // geometry
  unsigned int n_subdivisions_1D;
  double       geometry_left;
  double       geometry_right;

  // mass matrix
  double ghost_parameter_M = -1.0;

  // stiffness matrix
  double ghost_parameter_A = -1.0;

  // time stepping
  std::shared_ptr<Function<dim>> exact_solution;
  std::shared_ptr<Function<dim>> exact_solution_der;
  double                         start_t;
  double                         end_t;
  double                         cfl;

  // linear solver
  std::string  solver_name           = "ILU";
  unsigned int solver_max_iterations = 1000;
  double       solver_abs_tolerance  = 1.e-20;
  double       solver_rel_tolerance  = 1.e-14;

  // advection
  std::shared_ptr<Function<dim>> advection;

  // level set field
  unsigned int                   level_set_fe_degree;
  std::shared_ptr<Function<dim>> level_set_function;

  // output
};
