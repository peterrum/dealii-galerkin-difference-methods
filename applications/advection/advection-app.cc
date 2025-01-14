// Solve cut advection problem (GDM).
//
// Setup as in:
// DoD stabilization for higher-order advection in two dimensions
// by Florian Streitbürger, Gunnar Birke, Christian Engwer, Sandra May

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/time_stepping.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <gdm/advection/problem.h>

#include <fstream>

using namespace dealii;

template <int dim, typename Number = double>
class ExactSolution : public dealii::Function<dim, Number>
{
public:
  ExactSolution()
  {}

  virtual double
  value(const dealii::Point<dim> &p, const unsigned int = 1) const override
  {
    return std::max(0.0, 0.3 - p.distance(Point<dim>(-0.3, -0.3)));
  }

private:
};

template <int dim, typename Number = double>
class ExactSolutionDerivative : public dealii::Function<dim, Number>
{
public:
  ExactSolutionDerivative()
  {}

  virtual double
  value(const dealii::Point<dim> &p, const unsigned int = 1) const override
  {
    (void)p;

    return 0.0;
  }

private:
};



template <int dim>
void
test(ConvergenceTable &table)
{
  const double factor          = 27.0;
  const double factor_rotation = 0.0;

  const double increment  = 5.0;
  const double rotation_0 = increment * factor;
  const double rotation_1 = increment * (factor + factor_rotation);
  const double phi        = (numbers::PI * increment / 180.0) * factor; // TODO
  const double x_shift    = 0.25;

  Parameters<dim> params;

  // gerneral settings
  params.fe_degree    = 5;
  params.n_components = 1;
  params.composite    = true;

  // geometry
  params.n_subdivisions_1D = 200;
  params.geometry_left     = -1.0;
  params.geometry_right    = +1.0;

  // mass matrix
  params.ghost_parameter_M = 0.5;

  // stiffness matrix
  params.ghost_parameter_A = 0.5;

  // time stepping
  params.start_t = 0.0;
  params.end_t   = 0.6;
  params.cfl     = 0.2;

  params.exact_solution     = std::make_shared<ExactSolution<dim>>();
  params.exact_solution_der = std::make_shared<ExactSolutionDerivative<dim>>();

  params.max_val = 4.0;

  dealii::Tensor<1, dim> advection;
  advection[0] = 3.0;
  advection[1] = 1.0;

  params.advection = std::make_shared<Functions::ConstantFunction<dim, double>>(
    advection.begin_raw(), dim);

  advection[0] = 1.0;
  advection[1] = 2.0;

  params.advection_1 =
    std::make_shared<Functions::ConstantFunction<dim, double>>(
      advection.begin_raw(), dim);

  params.level_set_fe_degree = 1;
  const Point<dim> point     = {x_shift, 0.0};
  Tensor<1, dim>   normal;
  normal[0] = +std::sin(phi);
  normal[1] = -std::cos(phi);
  params.level_set_function =
    std::make_shared<Functions::SignedDistance::Plane<dim>>(point, normal);

  table.add_value("rot_0", rotation_0);
  table.add_value("rot_1", rotation_1);

  AdvectionProblem<dim> problem(params);
  problem.run(table);
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const std::string case_name = "parallel-convergence";

  ConvergenceTable table;

  test<2>(table);
}
