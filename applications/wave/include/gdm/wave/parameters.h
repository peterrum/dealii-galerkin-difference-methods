#pragma once

#include <deal.II/base/function.h>

using namespace dealii;

template <unsigned int dim>
struct Parameters
{
  // general settings
  std::string  simulation_type;
  unsigned int fe_degree;
  unsigned int n_components;

  // geometry
  unsigned int n_subdivisions_1D;
  double       geometry_left;
  double       geometry_right;

  std::function<Point<dim>(const Point<dim>)> mapping_q_cache_function;

  // mass matrix
  double ghost_parameter_M = -1.0;

  // stiffness matrix
  double                         ghost_parameter_A = -1.0;
  double                         nitsche_parameter = -1.0;
  std::shared_ptr<Function<dim>> function_interface_dbc;
  std::shared_ptr<Function<dim>> function_rhs;

  // time stepping
  std::shared_ptr<Function<dim>> exact_solution;
  double                         start_t;
  double                         end_t;
  double                         cfl;
  double                         cfl_pow;

  // linear solver
  std::string  solver_name           = "AMG";
  unsigned int solver_max_iterations = 1000;
  double       solver_abs_tolerance  = 1.e-20;
  double       solver_rel_tolerance  = 1.e-14;

  // level set field
  unsigned int                   level_set_fe_degree;
  std::shared_ptr<Function<dim>> level_set_function;

  // output
  unsigned int output_fe_degree = 1;
};
