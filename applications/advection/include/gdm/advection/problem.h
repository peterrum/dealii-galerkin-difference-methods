#pragma once

#include <gdm/advection/discretization.h>
#include <gdm/advection/mass.h>
#include <gdm/advection/parameters.h>
#include <gdm/advection/stiffness.h>
#include <gdm/data_out.h>
#include <gdm/system.h>
#include <gdm/vector_tools.h>

using namespace dealii;

template <int dim, typename Number = double>
class AdvectionProblem
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  AdvectionProblem(const Parameters<dim> &params)
    : comm(MPI_COMM_WORLD)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0) && false)
    , params(params)
    , discretization()
    , mass_matrix_operator(discretization)
    , mass_matrix_operator_opt(discretization,
                               NonMatching::LocationToLevelSet::outside)
    , stiffness_matrix_operator(discretization)
  {}

  void
  run(ConvergenceTable &table)
  {
    discretization.reinit(params);
    mass_matrix_operator.reinit(params);
    stiffness_matrix_operator.reinit(params);

    std::array<double, 6> error{};

    if (params.composite == false)
      {
        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double max_vel = 2.0;
        const double delta_t = discretization.get_dx() * params.cfl / max_vel;

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();

        // Initialize vectors
        BlockVectorType solution;
        stiffness_matrix_operator.initialize_dof_vector(solution);
        this->set_initial_condition(solution.block(1));

        // Setup solver
        this->setup_solver(mass_matrix);

        // helper function to evaluate right-hand-side vector
        const auto fu_rhs = [&](const double           time,
                                const BlockVectorType &stage_bc_and_solution) {
          BlockVectorType result;
          result.reinit(stage_bc_and_solution);

          // du/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(result,
                                                stage_bc_and_solution,
                                                time);

          const auto vec_rhs = result.block(1);
          this->solve(mass_matrix, result.block(1), vec_rhs);

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<BlockVectorType> rk;
        rk.initialize(runge_kutta_method);

        error = this->postprocess(0.0, solution.block(1));

        while ((time.is_at_end() == false) && (error[2] < 1.0 /*TODO*/))
          {
            stiffness_matrix_operator.initialize_time_step(
              solution, time.get_current_time()); // evaluate bc

            rk.evolve_one_time_step(fu_rhs,
                                    time.get_current_time(),
                                    time.get_next_step_size(),
                                    solution);

            // output result
            error = this->postprocess(time.get_current_time() +
                                        time.get_next_step_size(),
                                      solution.block(1));

            time.advance_time();
          }
      }
    else
      {
        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double max_vel = 2.0;
        const double delta_t = discretization.get_dx() * params.cfl / max_vel;

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix_0 = mass_matrix_operator.get_sparse_matrix();

        mass_matrix_operator_opt.reinit(params);
        const auto &mass_matrix_1 =
          mass_matrix_operator_opt.get_sparse_matrix();

        // Initialize vectors
        BlockVectorType solution;
        stiffness_matrix_operator.initialize_dof_vector(solution);
        this->set_initial_condition(solution.block(1));
        this->set_initial_condition(solution.block(3));

        // Setup solver
        this->setup_solver(mass_matrix_0, 0);
        this->setup_solver(mass_matrix_1, 1);

        // helper function to evaluate right-hand-side vector
        const auto fu_rhs = [&](const double           time,
                                const BlockVectorType &stage_bc_and_solution) {
          BlockVectorType result;
          result.reinit(stage_bc_and_solution);

          // du/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(result,
                                                stage_bc_and_solution,
                                                time);

          auto vec_rhs = result.block(1);
          this->solve(mass_matrix_0, result.block(1), vec_rhs, 0);

          vec_rhs = result.block(3);
          this->solve(mass_matrix_0, result.block(3), vec_rhs, 1);

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<BlockVectorType> rk;
        rk.initialize(runge_kutta_method);

        error = this->postprocess(0.0,
                                  solution.block(1),
                                  NonMatching::LocationToLevelSet::inside);
        error = this->postprocess(0.0,
                                  solution.block(3),
                                  NonMatching::LocationToLevelSet::outside);

        while ((time.is_at_end() == false) && (error[2] < 1.0 /*TODO*/))
          {
            stiffness_matrix_operator.initialize_time_step(
              solution, time.get_current_time()); // evaluate bc

            rk.evolve_one_time_step(fu_rhs,
                                    time.get_current_time(),
                                    time.get_next_step_size(),
                                    solution);

            // output result
            error = this->postprocess(time.get_current_time() +
                                        time.get_next_step_size(),
                                      solution.block(1),
                                      NonMatching::LocationToLevelSet::inside);
            error = this->postprocess(time.get_current_time() +
                                        time.get_next_step_size(),
                                      solution.block(3),
                                      NonMatching::LocationToLevelSet::outside);

            time.advance_time();
          }
      }

    table.add_value("fe_degree", params.fe_degree);
    table.add_value("cfl", params.cfl);
    table.add_value("n_subdivision", params.n_subdivisions_1D);
    table.add_value("error_2", error[2]);
    table.set_scientific("error_2", true);
    table.add_value("error_1", error[1]);
    table.set_scientific("error_1", true);
    table.add_value("error_inf", error[0]);
    table.set_scientific("error_inf", true);
    table.add_value("error_2_face", error[5]);
    table.set_scientific("error_2_face", true);
    table.add_value("error_1_face", error[4]);
    table.set_scientific("error_1_face", true);
    table.add_value("error_inf_face", error[3]);
    table.set_scientific("error_inf_face", true);

    pcout << std::endl;
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

  std::array<double, 6>
  postprocess(const double                          time,
              const VectorType                     &solution,
              const NonMatching::LocationToLevelSet location =
                NonMatching::LocationToLevelSet::inside)
  {
    const NonMatching::LocationToLevelSet inverse_location =
      (location == NonMatching::LocationToLevelSet::inside) ?
        NonMatching::LocationToLevelSet::outside :
        NonMatching::LocationToLevelSet::inside;

    static unsigned int counter = 0;

    const auto        &mapping          = discretization.get_mapping();
    const auto        &system           = discretization.get_system();
    const unsigned int fe_degree_output = 2;
    const NonMatching::MeshClassifier<dim> &mesh_classifier =
      discretization.get_mesh_classifier();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const VectorType            &level_set = discretization.get_level_set();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    auto exact_solution = params.exact_solution;

    // compute error
    exact_solution->set_time(time);

    // compute error
    const Quadrature<1> &quadrature_1D_error =
      discretization.get_quadrature_1D();

    NonMatching::RegionUpdateFlags region_update_flags_error;
    if (location == NonMatching::LocationToLevelSet::inside)
      region_update_flags_error.inside =
        update_values | update_JxW_values | update_quadrature_points;
    else if (location == NonMatching::LocationToLevelSet::outside)
      region_update_flags_error.outside =
        update_values | update_JxW_values | update_quadrature_points;
    else
      AssertThrow(false, ExcNotImplemented());
    region_update_flags_error.surface =
      update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values_error(
      fe,
      quadrature_1D_error,
      region_update_flags_error,
      mesh_classifier,
      level_set_dof_handler,
      level_set);

    double local_error_Linf            = 0;
    double local_error_L1              = 0;
    double local_error_L2_squared      = 0;
    double local_error_Linf_face       = 0;
    double local_error_L1_face         = 0;
    double local_error_L2_face_squared = 0;

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

          const std::optional<FEValues<dim>> &fe_values =
            (location == NonMatching::LocationToLevelSet::inside) ?
              non_matching_fe_values_error.get_inside_fe_values() :
              non_matching_fe_values_error.get_outside_fe_values();

          const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
            &fe_surface_values =
              non_matching_fe_values_error.get_surface_fe_values();


          std::vector<types::global_dof_index> local_dof_indices(
            fe[0].dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          if (fe_values)
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
                    solution_values.at(q) - exact_solution->value(point);

                  local_error_L2_squared +=
                    Utilities::fixed_power<2>(error_at_point) *
                    fe_values->JxW(q);

                  local_error_L1 +=
                    std::abs(error_at_point) * fe_values->JxW(q);

                  local_error_Linf =
                    std::max(local_error_Linf, std::abs(error_at_point));
                }
            }

          if (fe_surface_values)
            {
              std::vector<double> solution_values(
                fe_surface_values->n_quadrature_points);
              fe_surface_values->get_function_values(solution,
                                                     local_dof_indices,
                                                     solution_values);

              for (const unsigned int q :
                   fe_surface_values->quadrature_point_indices())
                {
                  const Point<dim> &point =
                    fe_surface_values->quadrature_point(q);
                  const double error_at_point =
                    solution_values.at(q) - exact_solution->value(point);

                  local_error_L2_face_squared +=
                    Utilities::fixed_power<2>(error_at_point) *
                    fe_surface_values->JxW(q);

                  local_error_L1_face +=
                    std::abs(error_at_point) * fe_surface_values->JxW(q);

                  local_error_Linf_face =
                    std::max(local_error_Linf_face, std::abs(error_at_point));
                }
            }
        }

    const double error_Linf =
      Utilities::MPI::max(local_error_Linf, MPI_COMM_WORLD);

    const double error_L1 = Utilities::MPI::sum(local_error_L1, MPI_COMM_WORLD);

    const double error_L2 =
      std::sqrt(Utilities::MPI::sum(local_error_L2_squared, MPI_COMM_WORLD));

    const double error_Linf_face =
      Utilities::MPI::max(local_error_Linf_face, MPI_COMM_WORLD);

    const double error_L1_face =
      Utilities::MPI::sum(local_error_L1_face, MPI_COMM_WORLD);

    const double error_L2_face = std::sqrt(
      Utilities::MPI::sum(local_error_L2_face_squared, MPI_COMM_WORLD));

    if (pcout.is_active())
      printf("%5d %8.5f %14.8e %14.8e %14.8e\n",
             counter,
             time,
             error_L2,
             error_L1,
             error_Linf);

    // output result -> Paraview
    GDM::DataOut<dim> data_out(system, mapping, fe_degree_output);
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
                                  *exact_solution,
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

    std::string file_name = "solution_" + std::to_string(counter) + ".vtu";
    data_out.write_vtu_in_parallel(file_name);

    counter++;

    return {{error_Linf,
             error_L1,
             error_L2,
             error_Linf_face,
             error_L1_face,
             error_L2_face}};
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
