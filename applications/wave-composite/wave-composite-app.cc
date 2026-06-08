#include <deal.II/base/parameter_handler.h>

#include <boost/math/special_functions/bessel.hpp>

#include <gdm/wave/problem.h>

using namespace dealii;

template <unsigned int dim>
void
fill_parameters(Parameters<dim>    &params,
                const std::string  &simulation_name,
                const unsigned int &domain_index,
                const unsigned int &fe_degree,
                const unsigned int &nb_division,
                const double       &cfl)
{
  params.domain_index      = domain_index;
  params.fe_degree         = fe_degree;
  params.n_subdivisions_1D = nb_division;
  params.cfl               = cfl;
  if (simulation_name == "step85")
    {
      // adopted from:
      // Simon Sticko, 2022, "deal.II: tutorial step-85"
      //
      // https://www.dealii.org/developer/doxygen/deal.II/step_85.html

      // general settings
      params.simulation_type = "poisson";
      // params.fe_degree       = 1;
      params.n_components = 1;

      // geometry
      // params.n_subdivisions_1D = 16;
      params.geometry_left  = -1.21;
      params.geometry_right = +1.21;

      // mass matrix
      params.ghost_parameter_M = -1.0;

      // stiffness matrix
      params.ghost_parameter_A = 0.5;
      params.nitsche_parameter = 5.0 * params.fe_degree;
      params.function_interface_dbc =
        std::make_shared<Functions::ConstantFunction<dim>>(1.0);
      params.function_rhs =
        std::make_shared<Functions::ConstantFunction<dim>>(4.0);

      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));
      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));

      // time stepping
      params.exact_solution =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto &p) { return 1. - 2. / dim * (p.norm_square() - 1.); });

      params.function_domain_dbc = params.exact_solution;
      params.start_t             = 0.0;
      params.end_t               = 0.1;
      // params.cfl     = 0.3;
      params.cfl_pow = 1.0;

      // linear solvers
      params.solver_name = "AMG";

      // level set field
      params.level_set_fe_degree = params.fe_degree;
      params.level_set_functions.push_back(
        std::make_shared<Functions::SignedDistance::Sphere<dim>>());

      // output
      params.output_fe_degree = params.fe_degree;
    }
  else if (simulation_name == "heat" || simulation_name == "heat-rk" ||
           simulation_name == "heat-impl")
    {
      // adopted from:
      // Gustav Ludvigsson, Kyle R. Steffen, Simon Sticko, Siyang Wang,
      // Qing Xia, Yekaterina Epshteyn, and Gunilla Kreiss. 2018.
      // "High-order numerical methods for 2D parabolic problems in
      // single and composite domains."
      //
      // https://link.springer.com/article/10.1007/s10915-017-0637-y

      // general settings
      params.simulation_type = (simulation_name == "heat") ?
                                 std::string("heat-impl") :
                                 simulation_name; // "heat-rk" or "heat-impl"
      // params.fe_degree       = 3;
      params.n_components = 1;

      // geometry
      // params.n_subdivisions_1D = 16;
      params.geometry_left  = -1.21;
      params.geometry_right = +1.21;

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

      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));
      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));

      // time stepping
      params.exact_solution      = params.function_interface_dbc;
      params.function_domain_dbc = params.function_interface_dbc;
      const auto make_function_initial_condition =
        [](const unsigned int domain_idx) -> std::shared_ptr<Function<dim>> {
        return std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [domain_idx](const auto &p) -> double {
            switch (domain_idx)
              {
                case 0:
                  if (dim == 1)
                    {
                      return std::pow(p[0], 9.0);
                    }
                  else if (dim == 2)
                    {
                      return std::pow(p[0], 9.0) * std::pow(p[1], 8.0);
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                case 1:
                  if (dim == 1)
                    {
                      return std::pow(p[0], 9.0);
                    }
                  else if (dim == 2)
                    {
                      return std::pow(p[0], 9.0) * std::pow(p[1], 8.0);
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                default:
                  AssertThrow(false, ExcNotImplemented());
                  return 0.0;
              }
          });
      };
      params.function_initial_condition.push_back(
        make_function_initial_condition(0));
      params.function_initial_condition.push_back(
        make_function_initial_condition(1));

      params.start_t = 0.0;
      params.end_t   = 0.1;

      if (params.simulation_type == "heat-rk")
        {
          // params.cfl     = 0.1 / params.fe_degree / params.fe_degree;
          params.cfl_pow = 2.0;
        }
      else if (params.simulation_type == "heat-impl")
        {
          // params.cfl     = 0.3;
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
      params.level_set_functions.push_back(
        std::make_shared<Functions::SignedDistance::Sphere<dim>>());

      // output
      params.output_fe_degree = params.fe_degree;
    }
  else if (simulation_name == "heat-composite")
    {
      // adopted from: TODO

      // general settings
      params.simulation_type = "heat-rk";
      // params.fe_degree       = 3;
      params.n_components = 1;
      params.composite    = true;

      // geometry
      // params.n_subdivisions_1D = 16;
      params.geometry_left  = -1.21;
      params.geometry_right = +1.21;

      // mass matrix
      params.ghost_parameter_M = 0.75;

      // stiffness matrix
      params.ghost_parameter_A = 1.5;
      params.nitsche_parameter = 5.0 * params.fe_degree;

      params.function_domain_dbc =
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

      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));
      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));

      // time stepping
      params.exact_solution = params.function_domain_dbc;
      const auto make_function_initial_condition =
        [](const unsigned int domain_idx) -> std::shared_ptr<Function<dim>> {
        return std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [domain_idx](const auto &p) -> double {
            switch (domain_idx)
              {
                case 0:
                  if (dim == 1)
                    {
                      return std::pow(p[0], 9.0);
                    }
                  else if (dim == 2)
                    {
                      return std::pow(p[0], 9.0) * std::pow(p[1], 8.0);
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                case 1:
                  if (dim == 1)
                    {
                      return std::pow(p[0], 9.0);
                    }
                  else if (dim == 2)
                    {
                      return std::pow(p[0], 9.0) * std::pow(p[1], 8.0);
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                default:
                  AssertThrow(false, ExcNotImplemented());
                  return 0.0;
              }
          });
      };
      params.function_initial_condition.push_back(
        make_function_initial_condition(0));
      params.function_initial_condition.push_back(
        make_function_initial_condition(1));

      params.start_t = 0.0;
      params.end_t   = 0.1;

      // params.cfl     = 0.1 / params.fe_degree / params.fe_degree;
      params.cfl_pow = 2.0;

      // linear solvers
      params.solver_name = "ILU";

      // level set field
      params.level_set_fe_degree = params.fe_degree;
      params.level_set_functions.push_back(
        std::make_shared<Functions::SignedDistance::Sphere<dim>>());

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
      // params.fe_degree       = 3;
      params.n_components = 1;

      // geometry
      // params.n_subdivisions_1D = 16;
      params.geometry_left  = -1.21;
      params.geometry_right = +1.21;

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
                // const auto wave_number = 3.0 * numbers::PI;
                // return boost::math::cyl_bessel_j(0, wave_number * r) *
                //        std::cos(wave_number * t);
                if (p[0] + p[1] < -1)
                  {
                    return ((std::cos(1.0) * std::cos(std::sqrt(10)) +
                             (std::sin(1.0) * std::sin(std::sqrt(10))) /
                               (std::sqrt(10))) *
                              std::cos(p[0] + p[1]) +
                            (-std::sin(1.0) * std::cos(std::sqrt(10)) +
                             (std::cos(1.0) * std::sin(std::sqrt(10))) /
                               (std::sqrt(10))) *
                              std::sin(p[0] + p[1])) *
                           std::cos(std::sqrt(2) * t);
                  }
                else if ((p[0] + p[1] >= -1) && (p[0] + p[1] <= 1))
                  {
                    return (std::cos(std::sqrt(10) * (p[0] + p[1]))) *
                           std::cos(std::sqrt(2) * t);
                  }
                else
                  {
                    return ((std::cos(1.0) * std::cos(std::sqrt(10)) +
                             (std::sin(1.0) * std::sin(std::sqrt(10))) /
                               (std::sqrt(10))) *
                              std::cos(p[0] + p[1]) +
                            (std::sin(1.0) * std::cos(std::sqrt(10)) -
                             (std::cos(1.0) * std::sin(std::sqrt(10))) /
                               (std::sqrt(10))) *
                              std::sin(p[0] + p[1])) *
                           std::cos(std::sqrt(2) * t);
                  }
              }
            else
              AssertThrow(false, ExcNotImplemented());
          });
      params.function_rhs = {};

      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));
      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(0.1));
      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));

      // time stepping
      params.exact_solution      = params.function_interface_dbc;
      params.function_domain_dbc = params.function_interface_dbc;

      const auto make_function_initial_condition =
        [](const unsigned int domain_idx) -> std::shared_ptr<Function<dim>> {
        return std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [domain_idx](const auto &p) -> double {
            const auto r = p.norm();
            switch (domain_idx)
              {
                case 0:
                  if (dim == 1)
                    {
                      const auto wave_number = 1.5 * numbers::PI;
                      return std::cos(wave_number * r);
                    }
                  else if (dim == 2)
                    {
                      // const auto wave_number = 3.0 * numbers::PI;
                      // return boost::math::cyl_bessel_j(0, wave_number * r);
                      return (std::cos(1.0) * std::cos(std::sqrt(10)) +
                              (std::sin(1.0) * std::sin(std::sqrt(10))) /
                                (std::sqrt(10))) *
                               std::cos(p[0] + p[1]) +
                             (-std::sin(1.0) * std::cos(std::sqrt(10)) +
                              (std::cos(1.0) * std::sin(std::sqrt(10))) /
                                (std::sqrt(10))) *
                               std::sin(p[0] + p[1]);
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                case 1:
                  if (dim == 1)
                    {
                      const auto wave_number = 1.5 * numbers::PI;
                      return std::cos(wave_number * r);
                    }
                  else if (dim == 2)
                    {
                      // const auto wave_number = 3.0 * numbers::PI;
                      // return boost::math::cyl_bessel_j(0, wave_number * r);
                      return std::cos(std::sqrt(10) * (p[0] + p[1]));
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                case 2:
                  if (dim == 1)
                    {
                      const auto wave_number = 1.5 * numbers::PI;
                      return std::cos(wave_number * r);
                    }
                  else if (dim == 2)
                    {
                      // const auto wave_number = 3.0 * numbers::PI;
                      // return boost::math::cyl_bessel_j(0, wave_number * r);
                      return (std::cos(1.0) * std::cos(std::sqrt(10)) +
                              (std::sin(1.0) * std::sin(std::sqrt(10))) /
                                (std::sqrt(10))) *
                               std::cos(p[0] + p[1]) +
                             (std::sin(1.0) * std::cos(std::sqrt(10)) -
                              (std::cos(1.0) * std::sin(std::sqrt(10))) /
                                (std::sqrt(10))) *
                               std::sin(p[0] + p[1]);
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                default:
                  AssertThrow(false, ExcNotImplemented());
                  return 0.0;
              }
          });
      };
      params.function_initial_condition.push_back(
        make_function_initial_condition(0));
      params.function_initial_condition.push_back(
        make_function_initial_condition(1));
      params.function_initial_condition.push_back(
        make_function_initial_condition(2));
      params.start_t = 0.0;
      params.end_t   = 2.0;
      // params.cfl            = 0.1;
      params.cfl_pow = 1.0;

      // linear solvers
      params.solver_name = "AMG";
      // params.solver_name = "direct";

      // level set field
      params.level_set_fe_degree = params.fe_degree;
      const auto make_level_set =
        [](const unsigned int domain_idx) -> std::shared_ptr<Function<dim>> {
        return std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [domain_idx](const auto &p) -> double {
            switch (domain_idx)
              {
                case 0:
                  return p[0] + p[1] + 1;
                case 1:
                  return std::max(-p[0] - p[1] - 1, p[0] + p[1] - 1);
                case 2:
                  return -p[0] - p[1] + 1;
                default:
                  AssertThrow(false, ExcNotImplemented());
                  return 0.0;
              }
          });
      };

      // params.level_set_functions.push_back(std::make_shared<Functions::SignedDistance::Sphere<dim>>());
      params.level_set_functions.push_back(make_level_set(0));
      params.level_set_functions.push_back(make_level_set(1));
      params.level_set_functions.push_back(make_level_set(2));

      // output
      params.output_fe_degree = params.fe_degree;
    }
  else if (simulation_name == "wave-composite")
    {
      // adopted from: TODO

      // general settings
      params.simulation_type = "wave-rk";
      // params.fe_degree       = 3;
      params.n_components = 1;
      params.composite    = true;

      // geometry
      // params.n_subdivisions_1D = 16;
      params.geometry_left  = -1.21;
      params.geometry_right = +1.21;
      // params.geometry_left     = -5;
      // params.geometry_right    = +5;

      // mass matrix
      params.ghost_parameter_M = 0.25 * std::sqrt(3.0);

      // stiffness matrix
      params.ghost_parameter_A = 0.50 * std::sqrt(3.0);
      params.nitsche_parameter = 5.0 * params.fe_degree;
      params.function_domain_dbc =
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
                return boost::math::cyl_bessel_j(0, wave_number * r) *
                       std::cos(wave_number * t);
                // if(p[0]+p[1]<-1)
                // {
                //     return
                //     ((std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(-std::sin(1.0)*std::cos(std::sqrt(10))+(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]))*std::cos(std::sqrt(2)*t);
                // }
                // else if ((p[0] + p[1] >= -1) && (p[0] + p[1] <= 1))
                // {
                //     return
                //     (std::cos(std::sqrt(10)*(p[0]+p[1])))*std::cos(std::sqrt(2)*t);
                // }
                // else
                // {
                //     return
                //     ((std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(std::sin(1.0)*std::cos(std::sqrt(10))-(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]))*std::cos(std::sqrt(2)*t);
                // }
              }
            else
              AssertThrow(false, ExcNotImplemented());
          });
      params.function_rhs = {};

      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));
      // params.speed.push_back(std::make_shared<Functions::ConstantFunction<dim>>(0.1));
      params.speed.push_back(
        std::make_shared<Functions::ConstantFunction<dim>>(1.0));

      // time stepping
      params.exact_solution = params.function_domain_dbc;

      const auto make_function_initial_condition =
        [](const unsigned int domain_idx) -> std::shared_ptr<Function<dim>> {
        return std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [domain_idx](const auto &p) -> double {
            const auto r = p.norm();
            switch (domain_idx)
              {
                case 0:
                  if (dim == 1)
                    {
                      const auto wave_number = 1.5 * numbers::PI;
                      return std::cos(wave_number * r);
                    }
                  else if (dim == 2)
                    {
                      const auto wave_number = 3.0 * numbers::PI;
                      return boost::math::cyl_bessel_j(0, wave_number * r);
                      // return
                      // (std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(-std::sin(1.0)*std::cos(std::sqrt(10))+(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]);
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                case 1:
                  if (dim == 1)
                    {
                      const auto wave_number = 1.5 * numbers::PI;
                      return std::cos(wave_number * r);
                    }
                  else if (dim == 2)
                    {
                      const auto wave_number = 3.0 * numbers::PI;
                      return boost::math::cyl_bessel_j(0, wave_number * r);
                      // return std::cos(std::sqrt(10)*(p[0]+p[1]));
                    }
                  else
                    AssertThrow(false, ExcNotImplemented());
                  // case 2:
                  //   if (dim == 1)
                  //   {
                  //     const auto wave_number = 1.5 * numbers::PI;
                  //     return std::cos(wave_number * r);
                  //   }
                  //   else if (dim == 2)
                  //     {
                  //       // const auto wave_number = 3.0 * numbers::PI;
                  //       // return boost::math::cyl_bessel_j(0, wave_number *
                  //       r); return
                  //       (std::cos(1.0)*std::cos(std::sqrt(10))+(std::sin(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::cos(p[0]+p[1])+(std::sin(1.0)*std::cos(std::sqrt(10))-(std::cos(1.0)*std::sin(std::sqrt(10)))/(std::sqrt(10)))*std::sin(p[0]+p[1]);
                  //     }
                  //   else
                  //     AssertThrow(false, ExcNotImplemented());
                default:
                  AssertThrow(false, ExcNotImplemented());
                  return 0.0;
              }
          });
      };
      params.function_initial_condition.push_back(
        make_function_initial_condition(0));
      params.function_initial_condition.push_back(
        make_function_initial_condition(1));
      // params.function_initial_condition.push_back(make_function_initial_condition(2));

      params.start_t = 0.0;
      params.end_t   = 2.0;
      // params.end_t          = (2 * numbers::PI )/ std::sqrt(2.0);
      // params.cfl            = 0.1;
      params.cfl_pow = 1.0;

      // linear solvers
      params.solver_name = "AMG";

      // level set field
      params.level_set_fe_degree = params.fe_degree;

      const auto make_level_set =
        [](const unsigned int domain_idx) -> std::shared_ptr<Function<dim>> {
        return std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [domain_idx](const auto &p) -> double {
            switch (domain_idx)
              {
                case 0:
                  return p[0] + p[1] + 1;
                case 1:
                  return std::max(-p[0] - p[1] - 1, p[0] + p[1] - 1);
                case 2:
                  return -p[0] - p[1] + 1;
                default:
                  AssertThrow(false, ExcNotImplemented());
                  return 0.0;
              }
          });
      };

      params.level_set_functions.push_back(
        std::make_shared<Functions::SignedDistance::Sphere<dim>>());
      // params.level_set_functions.push_back(make_level_set(0));
      // params.level_set_functions.push_back(make_level_set(1));
      // params.level_set_functions.push_back(make_level_set(2));

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

  if ((argc != 7) && ((argc != 2) || (std::string(argv[1]) == "--help")))
    {
      std::cout << "Usage: ./wave-app dim simulation" << std::endl;
      std::cout << std::endl;
      std::cout << "dim         number of dimensions (1-3)" << std::endl;
      std::cout << "simulation  name of simulation (step85, heat, wave)"
                << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;

      std::cout << "Usage: ./wave-app file" << std::endl;
      std::cout << std::endl;
      std::cout << "file        name of parameter file (*.json)" << std::endl;
      std::cout << std::endl;

      return 1;
    }

  // unsigned int dim;
  // std::string  simulation_name;
  // unsigned int domain_index;
  // unsigned int fe_degree;
  // unsigned int nb_division;
  // double cfl;
  unsigned int dim             = 1;
  std::string  simulation_name = "";
  unsigned int domain_index    = 0;
  unsigned int fe_degree       = 1;
  unsigned int nb_division     = 1;
  double       cfl             = 0.1;

  if (argc == 7)
    {
      dim             = std::atoi(argv[1]);
      simulation_name = std::string(argv[2]);
      domain_index    = std::atoi(argv[3]);
      fe_degree       = std::atoi(argv[4]);
      nb_division     = std::atoi(argv[5]);
      cfl             = std::atof(argv[6]);
    }
  else if (argc == 2)
    {
      dealii::ParameterHandler prm;
      prm.add_parameter("simulation name", simulation_name);
      prm.add_parameter("dim", dim);
      prm.add_parameter("domain index", domain_index);
      prm.add_parameter("fe_degree", fe_degree);
      prm.add_parameter("nb_division", nb_division);
      prm.add_parameter("cfl", cfl);
      prm.parse_input(std::string(argv[1]), "", true);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  if (dim == 1)
    {
      Parameters<1> params;
      fill_parameters(
        params, simulation_name, domain_index, fe_degree, nb_division, cfl);
      WaveProblem<1>(params).run();
    }
  else if (dim == 2)
    {
      Parameters<2> params;
      fill_parameters(
        params, simulation_name, domain_index, fe_degree, nb_division, cfl);
      WaveProblem<2>(params).run();
    }
  else
    AssertThrow(false, ExcNotImplemented());
}
