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
  ExactSolution(const double x_shift, const double phi, const double phi_add)
    : x_shift(x_shift)
    , phi(phi)
  {
    advection[0] = 2.0 * std::cos(phi + phi_add);
    advection[1] = 2.0 * std::sin(phi + phi_add);
  }

  virtual double
  value(const dealii::Point<dim> &p, const unsigned int = 1) const override
  {
    double                       t        = this->get_time();
    const dealii::Tensor<1, dim> position = p - t * advection;

    const double x_hat =
      std::cos(phi) * (position[0] - x_shift) + std::sin(phi) * position[1];

    return std::sin(std::sqrt(2.0) * numbers::PI * x_hat / (1.0 - x_shift));
  }

  const dealii::Tensor<1, dim> &
  get_transport_direction() const
  {
    return advection;
  }

private:
  dealii::Tensor<1, dim> advection;
  const double           x_shift;
  const double           phi;
};

template <int dim, typename Number = double>
class ExactSolutionDerivative : public dealii::Function<dim, Number>
{
public:
  ExactSolutionDerivative(const double x_shift,
                          const double phi,
                          const double phi_add)
    : x_shift(x_shift)
    , phi(phi)
  {
    advection[0] = 2.0 * std::cos(phi + phi_add);
    advection[1] = 2.0 * std::sin(phi + phi_add);
  }

  virtual double
  value(const dealii::Point<dim> &p, const unsigned int = 1) const override
  {
    double                       t        = this->get_time();
    const dealii::Tensor<1, dim> position = p - t * advection;

    const double x_hat =
      std::cos(phi) * (position[0] - x_shift) + std::sin(phi) * position[1];

    return std::cos(std::sqrt(2.0) * numbers::PI * x_hat / (1.0 - x_shift)) *
           (std::sqrt(2.0) * numbers::PI / (1.0 - x_shift)) *
           (std::cos(phi) * (-advection[0]) + std::sin(phi) * (-advection[1]));
  }

private:
  dealii::Tensor<1, dim> advection;
  const double           x_shift;
  const double           phi;
};



template <int dim>
void
test(ConvergenceTable  &table,
     const unsigned int fe_degree,
     const unsigned int n_subdivisions_1D,
     const double       cfl,
     const double       factor_rotation,
     const double       factor)
{
  const double increment  = 5.0;
  const double rotation_0 = increment * factor;
  const double rotation_1 = increment * (factor + factor_rotation);
  const double phi        = (numbers::PI * increment / 180.0) * factor; // TODO
  const double phi_add    = (numbers::PI * increment / 180.0) * factor_rotation;
  const double x_shift    = 0.2001;

  Parameters<dim> params;

  // gerneral settings
  params.fe_degree    = fe_degree;
  params.n_components = 1;

  // geometry
  params.n_subdivisions_1D = n_subdivisions_1D;
  params.geometry_left     = 0.0;
  params.geometry_right    = 1.0;

  // mass matrix
  params.ghost_parameter_M = 0.5;

  // stiffness matrix
  params.ghost_parameter_A = 0.5;

  // time stepping
  params.start_t = 0.0;
  params.end_t   = 0.1;
  params.cfl     = cfl;

  params.exact_solution =
    std::make_shared<ExactSolution<dim>>(x_shift, phi, phi_add);
  params.exact_solution_der =
    std::make_shared<ExactSolutionDerivative<dim>>(x_shift, phi, phi_add);

  dealii::Tensor<1, dim> advection;
  advection[0] = 2.0 * std::cos(phi + phi_add);
  advection[1] = 2.0 * std::sin(phi + phi_add);

  params.advection = std::make_shared<Functions::ConstantFunction<dim, double>>(
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

  // parallel ramp: fe degree, cfl, h
  if (case_name == "parallel-convergence")
    {
      const double factor = 5.0;

      for (const unsigned int fe_degree : {3, 5})
        {
          for (const double cfl : {0.4, 0.2, 0.1, 0.05, 0.025})
            {
              for (unsigned int n_subdivisions_1D = 10;
                   n_subdivisions_1D <= 100;
                   n_subdivisions_1D += 10)
                test<2>(table, fe_degree, n_subdivisions_1D, cfl, 0.0, factor);

              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                {
                  table.write_text(std::cout);
                  std::cout << std::endl;
                }

              table.clear();
            }
        }
    }

  // parallel ramp: fe degree, ramp degree
  if (case_name == "parallel-ramp-degree")
    {
      for (const unsigned int fe_degree : {3, 5})
        {
          for (unsigned int factor = 1.0; factor <= 9; ++factor)
            {
              const double       cfl = (fe_degree == 3) ? 0.4 : 0.1;
              const unsigned int n_subdivisions_1D = 40;

              test<2>(table, fe_degree, n_subdivisions_1D, cfl, 0.0, factor);
            }

          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              table.write_text(std::cout);
              std::cout << std::endl;
            }

          table.clear();
        }
    }

  // non-parallel ramp: fe degree, advection direction
  if (case_name == "non-parallel-advection-direction")
    {
      for (const unsigned int fe_degree : {3, 5})
        {
          for (int factor_rotation = 0.0; factor_rotation <= 18;
               ++factor_rotation)
            {
              const double       factor = 5;
              const double       cfl    = (fe_degree == 3) ? 0.4 : 0.1;
              const unsigned int n_subdivisions_1D = 40;

              test<2>(table,
                      fe_degree,
                      n_subdivisions_1D,
                      cfl,
                      factor_rotation - factor,
                      factor);
            }

          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              table.write_text(std::cout);
              std::cout << std::endl;
            }

          table.clear();
        }
    }
}
