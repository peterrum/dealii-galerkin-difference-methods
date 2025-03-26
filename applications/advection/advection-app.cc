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
#include <deal.II/base/parameter_handler.h>
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



void
test_1D(const std::string simulation_name, ConvergenceTable &table)
{
  const unsigned int dim = 1;

  if (simulation_name == "pipe-aligned")
    {
      Parameters<dim> params;

      // gerneral settings
      params.fe_degree    = 5;
      params.n_components = 1;
      params.composite    = false;

      params.solver_rel_tolerance = 1e-8;

      // geometry
      params.n_subdivisions_1D = 50;
      params.geometry_left     = -1.0;
      params.geometry_right    = +1.0;

      // mass matrix
      params.ghost_parameter_M = 0.5;

      // stiffness matrix
      params.ghost_parameter_A = 0.5;

      // time stepping
      params.start_t   = 0.0;
      params.end_t     = numbers::PI / 4.0;
      params.cfl       = 0.4;
      params.rk_method = RungeKuttaMethod::RK_AUTO;

      params.exact_solution =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            auto pp = p;
            pp[0] -= t;

            return std::exp(-16 * std::pow(pp.distance(Point<dim>(-0.3)), 2.0));
          });

      params.exact_solution_der =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) { return 0.0; });

      params.max_val = 1.0;

      params.advection = std::make_shared<FunctionFromFunctionObjects<dim>>(
        [](const auto &p, const unsigned int c) { return 1; }, dim);

      params.level_set_fe_degree = 1;
      params.level_set_function =
        std::make_shared<Functions::SignedDistance::Sphere<dim>>(Point<dim>{},
                                                                 1.2);

      AdvectionProblem<dim> problem(params);
      problem.run(table);
    }
  else if (simulation_name == "pipe-cut")
    {
      Parameters<dim> params;

      // gerneral settings
      params.fe_degree    = 5;
      params.n_components = 1;
      params.composite    = false;

      params.solver_rel_tolerance = 1e-8;

      const unsigned int n_subdivisions_1D = 50;
      const double       alpha             = 1e-3;
      const unsigned int n_ghost_cells     = 4;

      // geometry
      params.n_subdivisions_1D = n_subdivisions_1D + n_ghost_cells;
      params.geometry_left  = -1.0 - (2.0 / n_subdivisions_1D) * n_ghost_cells;
      params.geometry_right = +1.0;

      // mass matrix
      params.ghost_parameter_M = 0.5;

      // stiffness matrix
      params.ghost_parameter_A = 0.5;

      // time stepping
      params.start_t   = 0.0;
      params.end_t     = numbers::PI / 4.0;
      params.cfl       = 0.4;
      params.rk_method = RungeKuttaMethod::RK_AUTO;

      params.exact_solution =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            auto pp = p;
            pp[0] -= t;

            return std::exp(-16 * std::pow(pp.distance(Point<dim>(-0.3)), 2.0));
          });

      params.exact_solution_der =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) { return 0.0; });

      params.max_val = 1.0;

      params.advection = std::make_shared<FunctionFromFunctionObjects<dim>>(
        [](const auto &p, const unsigned int c) { return 1; }, dim);

      params.level_set_fe_degree = 1;

      const Point<dim> point(-1.0 - 2.0 / n_subdivisions_1D * alpha);
      Tensor<1, dim>   normal;
      normal[0] = -1;

      params.level_set_function =
        std::make_shared<Functions::SignedDistance::Plane<dim>>(point, normal);

      AdvectionProblem<dim> problem(params);
      problem.run(table);
    }
}



template <int dim>
void
test(const std::string simulation_name, ConvergenceTable &table)
{
  if (simulation_name == "composite_domain")
    {
      const double factor          = 27.0;
      const double factor_rotation = 0.0;

      const double increment  = 5.0;
      const double rotation_0 = increment * factor;
      const double rotation_1 = increment * (factor + factor_rotation);
      const double phi     = (numbers::PI * increment / 180.0) * factor; // TODO
      const double x_shift = 0.25;

      Parameters<dim> params;

      // gerneral settings
      params.fe_degree    = 5;
      params.n_components = 1;
      params.composite    = true;

      // geometry
      params.n_subdivisions_1D = 50;
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

      params.exact_solution =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [&](const auto t, const auto &p) {
            dealii::Tensor<1, dim> advection_0;
            advection_0[0] = 3.0;
            advection_0[1] = 1.0;

            dealii::Tensor<1, dim> advection_1;
            advection_1[0] = 1.0;
            advection_1[1] = 2.0;

            Tensor<1, dim> tangent;
            tangent[0] = +std::cos(phi);
            tangent[1] = +std::sin(phi);

            Vector<double>     rhs(2);
            Vector<double>     sol(2);
            FullMatrix<double> matrix(2, 2);
            matrix(0, 0) = -advection_1[0];
            matrix(1, 0) = -advection_1[1];
            matrix(0, 1) = +tangent[0];
            matrix(1, 1) = +tangent[1];
            matrix.gauss_jordan();

            rhs[0] = p[0] - x_shift;
            rhs[1] = p[1] - 0.0;
            matrix.vmult(sol, rhs);

            const double alpha = std::max(-sol[0], 0.0);
            const double t_1   = std::min(alpha, t);
            const double t_0   = t - t_1;

            const auto pp = p - t_0 * advection_0 - t_1 * advection_1;

            return std::exp(-16 *
                            std::pow(pp.distance(Point<dim>(-0.3, -0.3)), 2.0));
          });
      params.exact_solution_der =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto &) { return 0.0; });

      params.max_val = 4.0;

      dealii::Tensor<1, dim> advection;
      advection[0] = 3.0;
      advection[1] = 1.0;

      params.advection =
        std::make_shared<Functions::ConstantFunction<dim, double>>(
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
  else if (simulation_name == "pipe")
    {
      Parameters<dim> params;

      // gerneral settings
      params.fe_degree    = 5;
      params.n_components = 1;
      params.composite    = false;

      // geometry
      params.n_subdivisions_1D = 50;
      params.geometry_left     = +0.0;
      params.geometry_right    = +1.5;

      // mass matrix
      params.ghost_parameter_M = 0.5;

      // stiffness matrix
      params.ghost_parameter_A = 0.5;

      // time stepping
      params.start_t   = 0.0;
      params.end_t     = numbers::PI / 4.0;
      params.cfl       = 0.4;
      params.rk_method = RungeKuttaMethod::RK_AUTO;

      params.exact_solution =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            const double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
            const double d = std::atan2(p[1], p[0]);

            const Point<dim> pp(r * std::sin(d - t), r * std::cos(d - t));

            double temp = 0.0;

            for (unsigned int deg = 0; deg < 360; deg += 45)
              {
                const double d0 = deg * 2 * numbers::PI / 360.0;
                const double r0 = (deg % 90 == 0) ? 1.0 : 1.43;

                const double x0 = r0 * std::cos(d0);
                const double y0 = r0 * std::sin(d0);

                // temp += std::exp(-4 * pp.distance(Point<dim>(x0, y0)));
                temp += std::exp(
                  -4.0 * std::sqrt(std::pow(r * std::sin(d - t) - x0, 2.0) +
                                   std::pow(r * std::cos(d - t) - y0, 2.0)));
              }

            return temp;
          });

      params.exact_solution_der =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            const double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
            const double d = std::atan2(p[1], p[0]);

            const Point<dim> pp(r * std::sin(d - t), r * std::cos(d - t));

            double temp = 0.0;

            for (unsigned int deg = 0; deg < 360; deg += 45)
              {
                const double d0 = deg * 2 * numbers::PI / 360.0;
                const double r0 = (deg % 90 == 0) ? 1.0 : 1.43;

                const double x0 = r0 * std::cos(d0);
                const double y0 = r0 * std::sin(d0);

                temp +=
                  -4 *
                  (-r * (r * std::sin(d - t) - x0) * std::cos(d - t) +
                   r * (r * std::cos(d - t) - y0) * std::sin(d - t)) *
                  std::exp(-4 * sqrt(std::pow(r * std::sin(d - t) - x0, 2.0) +
                                     std::pow(r * std::cos(d - t) - y0, 2.0))) /
                  std::sqrt(std::pow(r * std::sin(d - t) - x0, 2.0) +
                            std::pow(r * std::cos(d - t) - y0, 2.0));
              }

            return temp;
          });

      params.max_val = 1.43;

      params.advection = std::make_shared<FunctionFromFunctionObjects<dim>>(
        [](const auto &p, const unsigned int c) {
          if (c == 0)
            return -p[1];
          else
            return p[0];
        },
        dim);

      params.level_set_fe_degree = 1;
      params.level_set_function =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto &p) {
            const auto radius = std::abs(std::complex<double>(p[0], p[1]));

            return -(1.43 - 1.0) / 2.0 + std::abs(radius - (1.43 + 1.0) / 2.0);
          });

      AdvectionProblem<dim> problem(params);
      problem.run(table);
    }
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  ConvergenceTable table;

  unsigned int dim;
  std::string  simulation_name;

  if (argc > 1)
    {
      if (std::string(argv[1]).find(".json") == std::string::npos)
        {
          AssertDimension(argc, 3);

          dim             = std::atoi(argv[1]);
          simulation_name = std::string(argv[2]);
        }
      else
        {
          dealii::ParameterHandler prm;
          prm.add_parameter("simulation name", simulation_name);
          prm.add_parameter("dim", dim);
          prm.parse_input(std::string(argv[1]), "", true);
        }
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  if (dim == 1)
    {
      test_1D(simulation_name, table);
    }
  else if (dim == 2)
    {
      test<2>(simulation_name, table);
    }
  else
    AssertThrow(false, ExcNotImplemented());

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      table.write_text(std::cout);
      std::cout << std::endl;
    }
}
