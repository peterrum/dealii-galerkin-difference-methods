#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/time_stepping.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <gdm/data_out.h>
#include <gdm/vector_tools.h>
#include <gdm/wave/discretization.h>
#include <gdm/wave/mass.h>
#include <gdm/wave/stiffness.h>

using namespace dealii;

template <int dim, typename Number = double>
class WaveProblem
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  WaveProblem(const Parameters<dim> &params)
    : comm(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , params(params)
    , discretization()
    , mass_matrix_operator(discretization)
    , mass_matrix_operator_opt(discretization,
                               NonMatching::LocationToLevelSet::outside)
    , stiffness_matrix_operator(discretization)
  {}

  void
  run()
  {
    discretization.reinit(params);
    mass_matrix_operator.reinit(params);
    stiffness_matrix_operator.reinit(params);

    if (params.simulation_type == "poisson")
      {
        // Compute stiffness matrix
        const auto &stiffness_matrix =
          stiffness_matrix_operator.get_sparse_matrix();

        // Initialize vectors
        VectorType vec_rhs, vec_solution;
        discretization.initialize_dof_vector(vec_rhs);
        discretization.initialize_dof_vector(vec_solution);

        // Compute right-hand-side vector
        stiffness_matrix_operator.compute_rhs(vec_rhs,
                                              vec_solution,
                                              false,
                                              0.0);

        // setup solver
        this->setup_solver(stiffness_matrix);

        // solve
        this->solve(stiffness_matrix, vec_solution, vec_rhs);

        // postprocess
        this->postprocess(0.0, vec_solution);
      }
    else if (params.simulation_type == "heat-rk")
      {
        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double delta_t =
          params.cfl * std::pow(discretization.get_dx(), params.cfl_pow);

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();

        // Initialize vectors
        VectorType vec_solution;
        discretization.initialize_dof_vector(vec_solution);
        this->set_initial_condition(vec_solution);

        // Setup solver
        this->setup_solver(mass_matrix);

        const auto fu_rhs = [&](const double time, const VectorType &solution) {
          VectorType result, vec_rhs;
          result.reinit(solution);
          vec_rhs.reinit(solution);

          // du/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(vec_rhs, solution, true, time);
          this->solve(mass_matrix, result, vec_rhs);

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<VectorType> rk;
        rk.initialize(runge_kutta_method);

        this->postprocess(0.0, vec_solution);

        while ((time.is_at_end() == false))
          {
            rk.evolve_one_time_step(fu_rhs,
                                    time.get_current_time(),
                                    time.get_next_step_size(),
                                    vec_solution);

            discretization.get_affine_constraints().distribute(vec_solution);

            this->postprocess(time.get_current_time() +
                                time.get_next_step_size(),
                              vec_solution);

            time.advance_time();
          }
      }
    else if (params.simulation_type == "heat-rk")
      {
        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double delta_t =
          params.cfl * std::pow(discretization.get_dx(), params.cfl_pow);

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix_0 = mass_matrix_operator.get_sparse_matrix();

        mass_matrix_operator_opt.reinit(params);
        const auto &mass_matrix_1 =
          mass_matrix_operator_opt.get_sparse_matrix();

        // Initialize vectors
        BlockVectorType vec_solution(2);
        discretization.initialize_dof_vector(vec_solution.block(0));
        discretization.initialize_dof_vector(vec_solution.block(1));
        this->set_initial_condition(vec_solution.block(0));
        this->set_initial_condition(vec_solution.block(1));

        // Setup solver
        this->setup_solver(mass_matrix_0, 0);
        this->setup_solver(mass_matrix_1, 0);

        const auto fu_rhs = [&](const double           time,
                                const BlockVectorType &solution) {
          BlockVectorType result, vec_rhs;
          result.reinit(solution);
          vec_rhs.reinit(solution);

          // du/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(vec_rhs, solution, true, time);
          this->solve(mass_matrix_0, result.block(0), vec_rhs.block(0), 0);
          this->solve(mass_matrix_1, result.block(1), vec_rhs.block(1), 1);

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<BlockVectorType> rk;
        rk.initialize(runge_kutta_method);

        this->postprocess(0.0,
                          vec_solution.block(0),
                          NonMatching::LocationToLevelSet::inside);
        this->postprocess(0.0,
                          vec_solution.block(1),
                          NonMatching::LocationToLevelSet::outside);

        while ((time.is_at_end() == false))
          {
            rk.evolve_one_time_step(fu_rhs,
                                    time.get_current_time(),
                                    time.get_next_step_size(),
                                    vec_solution);

            discretization.get_affine_constraints().distribute(
              vec_solution.block(0));
            discretization.get_affine_constraints().distribute(
              vec_solution.block(1));

            this->postprocess(time.get_current_time() +
                                time.get_next_step_size(),
                              vec_solution.block(0),
                              NonMatching::LocationToLevelSet::inside);
            this->postprocess(time.get_current_time() +
                                time.get_next_step_size(),
                              vec_solution.block(1),
                              NonMatching::LocationToLevelSet::outside);

            time.advance_time();
          }
      }
    else if (params.simulation_type == "heat-impl")
      {
        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double delta_t =
          params.cfl * std::pow(discretization.get_dx(), params.cfl_pow);

        // Compute matrix (M + dt * S)
        const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();
        const auto &stiffness_matrix =
          stiffness_matrix_operator.get_sparse_matrix();

        TrilinosWrappers::SparseMatrix system_matrix;
        system_matrix.reinit(mass_matrix);
        system_matrix.add(1.0, mass_matrix);
        system_matrix.add(delta_t, stiffness_matrix);
        system_matrix.compress(VectorOperation::values::add);

        // Initialize vectors
        VectorType vec_solution;
        discretization.initialize_dof_vector(vec_solution);
        this->set_initial_condition(vec_solution);

        // Setup solver
        this->setup_solver(system_matrix);

        const auto fu_rhs = [&](const double time, const VectorType &solution) {
          VectorType vec_rhs;
          vec_rhs.reinit(solution);

          // du/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(vec_rhs, solution, false, time);
          return vec_rhs;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        this->postprocess(0.0, vec_solution);

        while ((time.is_at_end() == false))
          {
            if (delta_t != time.get_next_step_size())
              {
                // note: in the last time step, the time-step size might
                // change -> set up again matrix and solver
                system_matrix = 0.0;
                system_matrix.add(1.0, mass_matrix);
                system_matrix.add(time.get_next_step_size(), stiffness_matrix);
                system_matrix.compress(VectorOperation::values::add);
                this->setup_solver(system_matrix);
              }

            // u := (M + dt * S)\(M u + dt * f(t, u))
            auto vec_rhs =
              fu_rhs(time.get_current_time() + time.get_next_step_size(),
                     vec_solution);
            vec_rhs *= time.get_next_step_size();
            mass_matrix.template vmult_add<VectorType>(vec_rhs, vec_solution);
            this->solve(system_matrix, vec_solution, vec_rhs);

            discretization.get_affine_constraints().distribute(vec_solution);

            this->postprocess(time.get_current_time() +
                                time.get_next_step_size(),
                              vec_solution);

            time.advance_time();
          }
      }
    else if (params.simulation_type == "wave-rk")
      {
        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double delta_t =
          params.cfl * std::pow(discretization.get_dx(), params.cfl_pow);

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();

        // Initialize vectors
        BlockVectorType vec_solution(2);
        discretization.initialize_dof_vector(vec_solution.block(0));
        discretization.initialize_dof_vector(vec_solution.block(1));
        this->set_initial_condition(vec_solution.block(0));

        // Setup solver
        this->setup_solver(mass_matrix);

        const auto fu_rhs = [&](const double           time,
                                const BlockVectorType &solution) {
          BlockVectorType result;
          result.reinit(solution);
          VectorType vec_rhs;
          vec_rhs.reinit(solution.block(0));

          // du/dt = v
          result.block(0) = solution.block(1);

          // dv/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(vec_rhs,
                                                solution.block(0),
                                                true,
                                                time);
          this->solve(mass_matrix, result.block(1), vec_rhs);

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<BlockVectorType> rk;
        rk.initialize(runge_kutta_method);

        this->postprocess(0.0, vec_solution.block(0));

        while ((time.is_at_end() == false))
          {
            rk.evolve_one_time_step(fu_rhs,
                                    time.get_current_time(),
                                    time.get_next_step_size(),
                                    vec_solution);

            discretization.get_affine_constraints().distribute(
              vec_solution.block(0));

            this->postprocess(time.get_current_time() +
                                time.get_next_step_size(),
                              vec_solution.block(0));

            time.advance_time();
          }
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

private:
  void
  set_initial_condition(VectorType &vector) const
  {
    params.exact_solution->set_time(params.start_t);

    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const GDM::System<dim>           &system  = discretization.get_system();

    GDM::VectorTools::interpolate(mapping,
                                  system,
                                  *params.exact_solution,
                                  vector);
  }

  void
  setup_solver(const TrilinosWrappers::SparseMatrix &sparse_matrix,
               const unsigned int                    id = 0)
  {
    if (params.solver_name == "AMG")
      preconditioner_amg[id].initialize(sparse_matrix);
    else if (params.solver_name == "ILU")
      preconditioner_ilu[id].initialize(sparse_matrix);
    else if (params.solver_name == "direct")
      solver_direct[id].initialize(sparse_matrix);
    else
      AssertThrow(false, ExcNotImplemented());
  }

  void
  solve(const TrilinosWrappers::SparseMatrix &sparse_matrix,
        VectorType                           &result,
        const VectorType                     &vec_rhs,
        const unsigned int                    id = 0)
  {
    if (params.solver_name == "AMG" || params.solver_name == "ILU")
      {
        ReductionControl solver_control(params.solver_max_iterations,
                                        params.solver_abs_tolerance,
                                        params.solver_rel_tolerance);

        SolverCG<VectorType> solver(solver_control);

        if (params.solver_name == "AMG")
          solver.solve(sparse_matrix, result, vec_rhs, preconditioner_amg[id]);
        else if (params.solver_name == "ILU")
          solver.solve(sparse_matrix, result, vec_rhs, preconditioner_ilu[id]);
        else
          AssertThrow(false, ExcNotImplemented());

        pcout << " [L] solved in " << solver_control.last_step() << std::endl;
      }
    else if (params.solver_name == "direct")
      {
        solver_direct[id].solve(sparse_matrix, result, vec_rhs);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

  void
  postprocess(const double                          time,
              const VectorType                     &solution,
              const NonMatching::LocationToLevelSet location =
                NonMatching::LocationToLevelSet::inside)
  {
    static std::array<unsigned int, 2> counter = {{0, 0}};

    auto &my_counter =
      counter[(location == NonMatching::LocationToLevelSet::inside) ? 0 : 1];

    const NonMatching::LocationToLevelSet inverse_location =
      (location == NonMatching::LocationToLevelSet::inside) ?
        NonMatching::LocationToLevelSet::outside :
        NonMatching::LocationToLevelSet::inside;

    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const Quadrature<1>              &quadrature_1D_error =
      discretization.get_quadrature_1D();
    const GDM::System<dim> &system = discretization.get_system();
    const NonMatching::MeshClassifier<dim> &mesh_classifier =
      discretization.get_mesh_classifier();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const VectorType            &level_set = discretization.get_level_set();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    // compute error
    params.exact_solution->set_time(time);

    NonMatching::RegionUpdateFlags region_update_flags_error;
    if (location == NonMatching::LocationToLevelSet::inside)
      region_update_flags_error.inside =
        update_values | update_JxW_values | update_quadrature_points;
    else if (location == NonMatching::LocationToLevelSet::outside)
      region_update_flags_error.outside =
        update_values | update_JxW_values | update_quadrature_points;
    else
      AssertThrow(false, ExcNotImplemented());

    NonMatching::FEValues<dim> non_matching_fe_values_error(
      fe,
      quadrature_1D_error,
      region_update_flags_error,
      mesh_classifier,
      level_set_dof_handler,
      level_set);

    double local_error_Linf       = 0;
    double local_error_L1         = 0;
    double local_error_L2_squared = 0;

    solution.update_ghost_values();
    for (const auto &cell : system.locally_active_cell_iterators())
      if (cell->is_locally_owned() &&
          (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
           inverse_location))
        {
          non_matching_fe_values_error.reinit(cell->dealii_iterator(),
                                              numbers::invalid_unsigned_int,
                                              numbers::invalid_unsigned_int,
                                              cell->active_fe_index());

          std::vector<types::global_dof_index> local_dof_indices(
            fe[0].dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          if (const std::optional<FEValues<dim>> &fe_values =
                (location == NonMatching::LocationToLevelSet::inside) ?
                  non_matching_fe_values_error.get_inside_fe_values() :
                  non_matching_fe_values_error.get_outside_fe_values())
            {
              std::vector<double> solution_values(
                fe_values->n_quadrature_points);
              fe_values->get_function_values(solution,
                                             local_dof_indices,
                                             solution_values);

              for (const unsigned int q : fe_values->quadrature_point_indices())
                {
                  const Point<dim> &point = fe_values->quadrature_point(q);
                  const double      error_at_point =
                    solution_values.at(q) - params.exact_solution->value(point);

                  local_error_L2_squared +=
                    Utilities::fixed_power<2>(error_at_point) *
                    fe_values->JxW(q);

                  local_error_L1 +=
                    std::abs(error_at_point) * fe_values->JxW(q);

                  local_error_Linf =
                    std::max(local_error_Linf, std::abs(error_at_point));
                }
            }
        }

    const double error_Linf =
      Utilities::MPI::max(local_error_Linf, MPI_COMM_WORLD);

    const double error_L1 = Utilities::MPI::sum(local_error_L1, MPI_COMM_WORLD);

    const double error_L2 =
      std::sqrt(Utilities::MPI::sum(local_error_L2_squared, MPI_COMM_WORLD));

    if (pcout.is_active())
      printf("%5d %8.5f %14.8e %14.8e %14.8e\n",
             my_counter,
             time,
             error_L2,
             error_L1,
             error_Linf);

    // output result -> Paraview
    GDM::DataOut<dim> data_out(system, mapping, params.output_fe_degree);
    solution.update_ghost_values();
    data_out.add_data_vector(solution, "solution");

    if (params.level_set_function)
      {
        VectorType level_set;
        discretization.initialize_dof_vector(level_set);
        GDM::VectorTools::interpolate(mapping,
                                      system,
                                      *params.level_set_function,
                                      level_set);
        level_set.update_ghost_values();
        data_out.add_data_vector(level_set, "level_set");
      }

    VectorType analytical_solution;
    discretization.initialize_dof_vector(analytical_solution);
    GDM::VectorTools::interpolate(mapping,
                                  system,
                                  *params.exact_solution,
                                  analytical_solution);
    analytical_solution.update_ghost_values();
    data_out.add_data_vector(analytical_solution, "analytical_solution");

    if (true)
      data_out.set_cell_selection(
        [&](const typename Triangulation<dim>::cell_iterator &cell) {
          return cell->is_active() && cell->is_locally_owned() &&
                 mesh_classifier.location_to_level_set(cell) !=
                   inverse_location;
        });

    data_out.build_patches();

    std::string file_name =
      std::string("solution_") +
      ((location == NonMatching::LocationToLevelSet::inside) ? "i_" : "o_") +
      std::to_string(my_counter) + ".vtu";
    data_out.write_vtu_in_parallel(file_name);

    my_counter++;
  }

  const MPI_Comm     comm;
  ConditionalOStream pcout;

  const Parameters<dim> &params;

  Discretization<dim, Number> discretization;

  MassMatrixOperator<dim, Number>      mass_matrix_operator;
  MassMatrixOperator<dim, Number>      mass_matrix_operator_opt;
  StiffnessMatrixOperator<dim, Number> stiffness_matrix_operator;

  std::array<TrilinosWrappers::PreconditionAMG, 2> preconditioner_amg;
  std::array<TrilinosWrappers::PreconditionILU, 2> preconditioner_ilu;
  std::array<TrilinosWrappers::SolverDirect, 2>    solver_direct;
};
