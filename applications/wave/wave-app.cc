#include <gdm/wave/wave-problem.h>

using namespace dealii;

template <unsigned int dim>
void
fill_parameters(Parameters<dim> &params, const std::string &simulation_name)
{
  if (simulation_name == "step85")
    {
      // adopted from:
      // Simon Sticko, 2022, "deal.II: tutorial step-85"
      //
      // https://www.dealii.org/developer/doxygen/deal.II/step_85.html

      // general settings
      params.simulation_type = "poisson";
      params.fe_degree       = 3;
      params.n_components    = 1;

      // geometry
      params.n_subdivisions_1D = 40;
      params.geometry_left     = -1.21;
      params.geometry_right    = +1.21;

      // mass matrix
      params.ghost_parameter_M = -1.0;

      // stiffness matrix
      params.ghost_parameter_A = 0.5;
      params.nitsche_parameter = 5.0 * params.fe_degree;
      params.function_interface_dbc =
        std::make_shared<Functions::ConstantFunction<dim>>(1.0);
      params.function_rhs =
        std::make_shared<Functions::ConstantFunction<dim>>(4.0);

      // time stepping
      params.exact_solution =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto &p) { return 1. - 2. / dim * (p.norm_square() - 1.); });

      params.start_t = 0.0;
      params.end_t   = 0.1;
      params.cfl     = 0.3;
      params.cfl_pow = 1.0;

      // linear solvers
      params.solver_name = "AMG";

      // level set field
      params.level_set_fe_degree = params.fe_degree;
      params.level_set_function =
        std::make_shared<Functions::SignedDistance::Sphere<dim>>();

      // output
      params.output_fe_degree = params.fe_degree;
    }
  else if (simulation_name == "heat")
    {
      // adopted from:
      // Gustav Ludvigsson, Kyle R. Steffen, Simon Sticko, Siyang Wang,
      // Qing Xia, Yekaterina Epshteyn, and Gunilla Kreiss. 2018.
      // "High-order numerical methods for 2D parabolic problems in
      // single and composite domains."
      //
      // https://link.springer.com/article/10.1007/s10915-017-0637-y

      // general settings
      params.simulation_type = "heat-impl"; // "heat-rk" or "heat-impl"
      params.fe_degree       = 3;
      params.n_components    = 1;

      // geometry
      params.n_subdivisions_1D = 40;
      params.geometry_left     = -1.21;
      params.geometry_right    = +1.21;

      // mass matrix
      params.ghost_parameter_M = 0.75;

      // stiffness matrix
      params.ghost_parameter_A = 1.5;
      params.nitsche_parameter = 5.0 * params.fe_degree;

      params.function_interface_dbc =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            if (dim == 1)
              return std::pow(p[0], 9.0) * std::exp(-t);
            else if (dim == 2)
              return std::pow(p[0], 9.0) * std::pow(p[1], 8.0) * std::exp(-t);

            AssertThrow(false, ExcNotImplemented());

            return 0.0;
          });

      params.function_rhs =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            if (dim == 1)
              return -std::pow(p[0], 7.0) * std::exp(-t) *
                     (std::pow(p[0], 2.0) + 72);
            else if (dim == 2)
              return -std::pow(p[0], 7.0) * std::pow(p[1], 6.0) * std::exp(-t) *
                     (std::pow(p[0], 2.0) * std::pow(p[1], 2.0) +
                      72 * std::pow(p[1], 2.0) + 56 * std::pow(p[0], 2.0));

            AssertThrow(false, ExcNotImplemented());

            return 0.0;
          });

      // time stepping
      params.exact_solution = params.function_interface_dbc;
      params.start_t        = 0.0;
      params.end_t          = 0.1;

      if (params.simulation_type == "heat-rk")
        {
          params.cfl     = 0.3 / params.fe_degree / params.fe_degree;
          params.cfl_pow = 2.0;
        }
      else if (params.simulation_type == "heat-impl")
        {
          params.cfl     = 0.3;
          params.cfl_pow = 1.0;
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      // linear solvers
      params.solver_name = "ILU";

      // level set field
      params.level_set_fe_degree = params.fe_degree;
      params.level_set_function =
        std::make_shared<Functions::SignedDistance::Sphere<dim>>();

      // output
      params.output_fe_degree = params.fe_degree;
    }
  else if (simulation_name == "wave")
    {
      // adopted from:
      // Simon Sticko, Gustav Ludvigsson, and Gunilla Kreiss. 2020.
      // "High-order cut finite elements for the elastic wave
      // equation."
      //
      // https://link.springer.com/article/10.1007/s10444-020-09785-z

      // general settings
      params.simulation_type = "wave-rk";
      params.fe_degree       = 3;
      params.n_components    = 1;

      // geometry
      params.n_subdivisions_1D = 40;
      params.geometry_left     = -1.21;
      params.geometry_right    = +1.21;

      // mass matrix
      params.ghost_parameter_M = 0.25 * std::sqrt(3.0);

      // stiffness matrix
      params.ghost_parameter_A = 0.50 * std::sqrt(3.0);
      params.nitsche_parameter = 5.0 * params.fe_degree;
      params.function_interface_dbc =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            const auto r = p.norm();

            if (dim == 1)
              {
                const auto wave_number = 1.5 * numbers::PI;
                return std::cos(wave_number * r) * std::cos(wave_number * t);
              }
            else if (dim == 2)
              {
                const auto wave_number = 3.0 * numbers::PI;
                return std::cyl_bessel_j(0, wave_number * r) *
                       std::cos(wave_number * t);
              }
            else
              AssertThrow(false, ExcNotImplemented());
          });
      params.function_rhs = {};

      // time stepping
      params.exact_solution = params.function_interface_dbc;
      params.start_t        = 0.0;
      params.end_t          = 2.0;
      params.cfl            = 0.3;
      params.cfl_pow        = 1.0;

      // linear solvers
      params.solver_name = "AMG";

      // level set field
      params.level_set_fe_degree = params.fe_degree;
      params.level_set_function =
        std::make_shared<Functions::SignedDistance::Sphere<dim>>();

      // output
      params.output_fe_degree = params.fe_degree;
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());

      // TODO: read from file
    }
}


/**
 * Run as:
 * ./tests/wave-app.debug/wave-app.debug 1 step85
 */
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc == 3, ExcInternalError());

  const unsigned int dim = std::atoi(argv[1]);
  const std::string  simulation_name(argv[2]);

  if (dim == 1)
    {
      Parameters<1> params;
      fill_parameters(params, simulation_name);
      WaveProblem<1>(params).run();
    }
  else if (dim == 2)
    {
      Parameters<2> params;
      fill_parameters(params, simulation_name);
      WaveProblem<2>(params).run();
    }
  else
    AssertThrow(false, ExcNotImplemented());
}
