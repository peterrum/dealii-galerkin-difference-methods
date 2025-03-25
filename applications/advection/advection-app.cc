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
          [](const auto &p) {
            return std::max(0.0, 0.3 - p.distance(Point<dim>(-0.3, -0.3)));
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
      params.start_t = 0.0;
      params.end_t   = 0.01;
      params.cfl     = 0.2;

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

                temp += std::exp(-4 * pp.distance(Point<dim>(x0, y0)));
              }

            return temp;
          });

      params.exact_solution_der =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            return 0.0;
            // return std::sqrt(p[0] * p[0] + p[1] * p[1]) *
            //        std::sin(std::atan2(p[1], p[0]) - t);
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

  std::string simulation_name;

  if (argc > 1)
    {
      if (std::string(argv[1]).find(".json") == std::string::npos)
        {
          simulation_name = std::string(argv[1]);
        }
      else
        {
          dealii::ParameterHandler prm;
          prm.add_parameter("simulation name", simulation_name);
          prm.parse_input(std::string(argv[1]), "", true);
        }
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  test<2>(simulation_name, table);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      table.write_text(std::cout);
      std::cout << std::endl;
    }
}
