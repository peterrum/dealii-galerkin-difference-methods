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
    , stiffness_matrix_operator(discretization)
  {}

  void
  reinit(const Parameters<dim> &params)
  {
    this->function_initial_condition    = params.function_initial_condition;
  }

  void
  run()
  {
    this->reinit(params);
    discretization.reinit(params);
    mass_matrix_operator.reinit(params);
    stiffness_matrix_operator.reinit(params);

    if (params.simulation_type == "poisson")
      {
        // Compute stiffness matrix
        const auto &stiffness_matrix =
          stiffness_matrix_operator.get_sparse_matrix();
        this->n_solvers();

        // Initialize vectors
        VectorType vec_rhs, vec_solution;
        discretization.initialize_dof_vector(vec_rhs,params.domain_index);
        discretization.initialize_dof_vector(vec_solution,params.domain_index);

        // Compute right-hand-side vector
        stiffness_matrix_operator.compute_rhs(params.domain_index,
                                              vec_rhs,
                                              vec_solution,
                                              false,
                                              0.0);

        // setup solver
        this->setup_solver(*stiffness_matrix[params.domain_index], params.domain_index);

        // solve
        this->solve(*stiffness_matrix[params.domain_index], vec_solution, vec_rhs, params.domain_index);

        // postprocess
        this->postprocess(0.0, vec_solution, params.domain_index);
      }
    else if ((params.simulation_type == "heat-rk") && (!params.composite))
      {
        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double delta_t =
          params.cfl * std::pow(discretization.get_dx(), params.cfl_pow);

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();
        this->n_solvers();

        // Initialize vectors
        VectorType vec_solution;
        discretization.initialize_dof_vector(vec_solution,params.domain_index);
        // this->set_initial_condition(vec_solution);
        this->set_initial_condition(vec_solution,params.domain_index);

        // Setup solver
        this->setup_solver(*mass_matrix[params.domain_index],params.domain_index);

        const auto fu_rhs = [&](const double time, const VectorType &solution) {
          VectorType result, vec_rhs;
          result.reinit(solution);
          vec_rhs.reinit(solution);

          // du/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(params.domain_index, vec_rhs, solution, true, time);
          this->solve(*mass_matrix[params.domain_index], result, vec_rhs, params.domain_index);

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<VectorType> rk;
        rk.initialize(runge_kutta_method);

        this->postprocess(0.0, vec_solution, params.domain_index);

        while ((time.is_at_end() == false))
          {
            rk.evolve_one_time_step(fu_rhs,
                                    time.get_current_time(),
                                    time.get_next_step_size(),
                                    vec_solution);

            discretization.get_affine_constraints().distribute(vec_solution);

            this->postprocess(time.get_current_time() +
                                time.get_next_step_size(),
                              vec_solution,
                            params.domain_index);

            time.advance_time();
          }
      }
    else if (params.simulation_type == "heat-rk")
      {
        AssertThrow(params.composite, ExcInternalError());

        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double delta_t =
          params.cfl * std::pow(discretization.get_dx(), params.cfl_pow);

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();
        this->n_solvers();

        // Initialize vectors
        BlockVectorType vec_solution(discretization.get_level_sets().size());
        for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
          discretization.initialize_dof_vector(vec_solution.block(dom_idx),dom_idx);
          // this->set_initial_condition(vec_solution.block(dom_idx));
          this->set_initial_condition(vec_solution.block(dom_idx),dom_idx);
          // Setup solver
          this->setup_solver(*mass_matrix[dom_idx], dom_idx);
        }

        const auto fu_rhs = [&](const double           time,
                                const BlockVectorType &solution) {
          BlockVectorType result, vec_rhs;
          result.reinit(solution);
          vec_rhs.reinit(solution);

          // du/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(vec_rhs, solution, true, time);
          for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
            this->solve(*mass_matrix[dom_idx], result.block(dom_idx), vec_rhs.block(dom_idx), dom_idx);
          }

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<BlockVectorType> rk;
        rk.initialize(runge_kutta_method);

        for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
          this->postprocess(0.0,
                          vec_solution.block(dom_idx),
                          dom_idx);
        }

        while ((time.is_at_end() == false))
          {
            rk.evolve_one_time_step(fu_rhs,
                                    time.get_current_time(),
                                    time.get_next_step_size(),
                                    vec_solution);

            for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
              discretization.get_affine_constraints().distribute(
                vec_solution.block(dom_idx));
              this->postprocess(time.get_current_time() +
                                  time.get_next_step_size(),
                                  vec_solution.block(dom_idx),
                                  dom_idx);
            }

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
        this->n_solvers();

        TrilinosWrappers::SparseMatrix system_matrix;
        system_matrix.reinit(*mass_matrix[params.domain_index]);
        system_matrix.add(1.0, *mass_matrix[params.domain_index]);
        system_matrix.add(delta_t, *stiffness_matrix[params.domain_index]);
        system_matrix.compress(VectorOperation::values::add);

        // Initialize vectors
        VectorType vec_solution;
        discretization.initialize_dof_vector(vec_solution,params.domain_index);
        // this->set_initial_condition(vec_solution);
        this->set_initial_condition(vec_solution,params.domain_index);

        // Setup solver
        this->setup_solver(system_matrix,params.domain_index);

        const auto fu_rhs = [&](const double time, const VectorType &solution) {
          VectorType vec_rhs;
          vec_rhs.reinit(solution);

          // du/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(params.domain_index, vec_rhs, solution, false, time);
          return vec_rhs;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        this->postprocess(0.0, vec_solution,params.domain_index);

        while ((time.is_at_end() == false))
          {
            if (delta_t != time.get_next_step_size())
              {
                // note: in the last time step, the time-step size might
                // change -> set up again matrix and solver
                system_matrix = 0.0;
                system_matrix.add(1.0, *mass_matrix[params.domain_index]);
                system_matrix.add(time.get_next_step_size(), *stiffness_matrix[params.domain_index]);
                system_matrix.compress(VectorOperation::values::add);
                this->setup_solver(system_matrix,params.domain_index);
              }

            // u := (M + dt * S)\(M u + dt * f(t, u))
            auto vec_rhs =
              fu_rhs(time.get_current_time() + time.get_next_step_size(),
                     vec_solution);
            vec_rhs *= time.get_next_step_size();
            mass_matrix[params.domain_index]->template vmult_add<VectorType>(vec_rhs, vec_solution);
            this->solve(system_matrix, vec_solution, vec_rhs, params.domain_index);

            discretization.get_affine_constraints().distribute(vec_solution);

            this->postprocess(time.get_current_time() +
                                time.get_next_step_size(),
                              vec_solution,params.domain_index);

            time.advance_time();
          }
      }
    else if ((params.simulation_type == "wave-rk") && (!params.composite))
      {
        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double delta_t =
          params.cfl * std::pow(discretization.get_dx(), params.cfl_pow);

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();
        this->n_solvers();

        // Initialize vectors
        BlockVectorType vec_solution(2);
        discretization.initialize_dof_vector(vec_solution.block(0),params.domain_index);
        discretization.initialize_dof_vector(vec_solution.block(1),params.domain_index);
        // this->set_initial_condition(vec_solution.block(0));
        this->set_initial_condition(vec_solution.block(0),params.domain_index);

        // Setup solver
        this->setup_solver(*mass_matrix[params.domain_index], params.domain_index);

        const auto fu_rhs = [&](const double           time,
                                const BlockVectorType &solution) {
          BlockVectorType result;
          result.reinit(solution);
          VectorType vec_rhs;
          vec_rhs.reinit(solution.block(0));

          // du/dt = v
          result.block(0) = solution.block(1);

          // dv/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(params.domain_index, vec_rhs,
                                                solution.block(0),
                                                true,
                                                time);
          this->solve(*mass_matrix[params.domain_index], result.block(1), vec_rhs, params.domain_index);

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<BlockVectorType> rk;
        rk.initialize(runge_kutta_method);

        this->postprocess(0.0, vec_solution.block(0), params.domain_index);

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
                              vec_solution.block(0),
                            params.domain_index);

            time.advance_time();
          }
      }
    else if (params.simulation_type == "wave-rk")
      {
        AssertThrow(params.composite, ExcInternalError());

        const double start_t = params.start_t;
        const double end_t   = params.end_t;
        const double delta_t =
          params.cfl * std::pow(discretization.get_dx(), params.cfl_pow);

        const TimeStepping::runge_kutta_method runge_kutta_method =
          TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

        // Compute mass matrix
        const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();
        this->n_solvers();

        // Initialize vectors
        BlockVectorType vec_solution(2 * discretization.get_level_sets().size());
        for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
          discretization.initialize_dof_vector(vec_solution.block(dom_idx),dom_idx);
          discretization.initialize_dof_vector(vec_solution.block(discretization.get_level_sets().size() + dom_idx),dom_idx);
          this->set_initial_condition(vec_solution.block(dom_idx),dom_idx);
          this->setup_solver(*mass_matrix[dom_idx], dom_idx);
        }

        const auto fu_rhs = [&](const double           time,
                                const BlockVectorType &solution) {
          BlockVectorType result;
          result.reinit(solution);
          BlockVectorType vec_rhs(discretization.get_level_sets().size());
          for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
            vec_rhs.block(dom_idx).reinit(solution.block(dom_idx));
            // du/dt = v
            result.block(dom_idx) = solution.block(discretization.get_level_sets().size()+dom_idx);
          }

          // dv/dt = f(t, u)
          stiffness_matrix_operator.compute_rhs(vec_rhs, solution, true, time);
          for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
            this->solve(*mass_matrix[dom_idx], result.block(discretization.get_level_sets().size()+dom_idx), vec_rhs.block(dom_idx), dom_idx);
          }

          return result;
        };

        // Perform time stepping
        DiscreteTime time(start_t, end_t, delta_t);

        TimeStepping::ExplicitRungeKutta<BlockVectorType> rk;
        rk.initialize(runge_kutta_method);
        for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
          this->postprocess(0.0,
                          vec_solution.block(dom_idx),
                          dom_idx);
        }                 

        while ((time.is_at_end() == false))
          {
            rk.evolve_one_time_step(fu_rhs,
                                    time.get_current_time(),
                                    time.get_next_step_size(),
                                    vec_solution);
            for(size_t dom_idx = 0; dom_idx < discretization.get_level_sets().size(); ++dom_idx){
              discretization.get_affine_constraints().distribute(
              vec_solution.block(dom_idx));

              this->postprocess(time.get_current_time() +
                                time.get_next_step_size(),
                              vec_solution.block(dom_idx),
                              dom_idx);
            }
            time.advance_time();
          }
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

private:
  std::vector<std::shared_ptr<Function<dim>>> function_initial_condition;
  void
  set_initial_condition(VectorType &vector, const unsigned int &dom_idx) const
  {
    params.exact_solution->set_time(params.start_t);

    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const GDM::System<dim>           &system  = discretization.get_system();

    GDM::VectorTools::interpolate(mapping,
                                  system,
                                  *function_initial_condition[dom_idx],
                                  vector);
  }

  void
  set_initial_condition(VectorType &vector, size_t &dom_idx) const
  {
    params.exact_solution->set_time(params.start_t);

    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const GDM::System<dim>           &system  = discretization.get_system();

    GDM::VectorTools::interpolate(mapping,
                                  system,
                                  *function_initial_condition[dom_idx],
                                  vector);
  }

  void n_solvers()
  {
      const unsigned int n = discretization.get_level_sets().size();
      
      if (params.solver_name == "AMG")
      {
          preconditioner_amg.resize(n);
          for (unsigned int i = 0; i < n; ++i)
              preconditioner_amg[i] = std::make_shared<TrilinosWrappers::PreconditionAMG>();
      }
      else if (params.solver_name == "ILU")
      {
          preconditioner_ilu.resize(n);
          for (unsigned int i = 0; i < n; ++i)
              preconditioner_ilu[i] = std::make_shared<TrilinosWrappers::PreconditionILU>();
      }
      else if (params.solver_name == "direct")
      {
          solver_direct.resize(n);
          for (unsigned int i = 0; i < n; ++i)
              solver_direct[i] = std::make_shared<TrilinosWrappers::SolverDirect>();
      }
      else
          AssertThrow(false, ExcNotImplemented());
  }

  void
  setup_solver(const TrilinosWrappers::SparseMatrix &sparse_matrix,
               const unsigned int                    id = 0)
  {
    if (params.solver_name == "AMG")
      preconditioner_amg[id]->initialize(sparse_matrix);
    else if (params.solver_name == "ILU")
      preconditioner_ilu[id]->initialize(sparse_matrix);
    else if (params.solver_name == "direct")
      solver_direct[id]->initialize(sparse_matrix);
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
          solver.solve(sparse_matrix, result, vec_rhs, *preconditioner_amg[id]);
        else if (params.solver_name == "ILU")
          solver.solve(sparse_matrix, result, vec_rhs, *preconditioner_ilu[id]);
        else
          AssertThrow(false, ExcNotImplemented());

        pcout << " [L] solved in " << solver_control.last_step() << std::endl;
      }
    else if (params.solver_name == "direct")
      {
        (*solver_direct[id]).solve(sparse_matrix, result, vec_rhs);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

  void
  postprocess(const double                          time,
              const VectorType                     &solution,
              const int                           domain_idx=0)
  {
    static std::array<unsigned int, 2> counter = {{0, 0}};

    auto &my_counter =
      counter[(domain_idx == 0) ? 0 : 1];

    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const Quadrature<1>              &quadrature_1D_error =
      discretization.get_quadrature_1D();
    const GDM::System<dim> &system = discretization.get_system();
    const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> &mesh_classifiers =
      discretization.get_mesh_classifiers();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const std::vector<VectorType>              &level_sets = discretization.get_level_sets();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    // compute error
    params.exact_solution->set_time(time);

    NonMatching::RegionUpdateFlags region_update_flags_error;
    region_update_flags_error.inside =
        update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values_error(
      fe,
      quadrature_1D_error,
      region_update_flags_error,
      *mesh_classifiers[domain_idx],
      level_set_dof_handler,
      level_sets[domain_idx]);

    double local_error_Linf       = 0;
    double local_error_L1         = 0;
    double local_error_L2_squared = 0;

    solution.update_ghost_values();
    for (const auto &cell : system.locally_active_cell_iterators())
      if (cell->is_locally_owned() &&
          (mesh_classifiers[domain_idx]->location_to_level_set(cell->dealii_iterator()) !=
           NonMatching::LocationToLevelSet::outside))
        {
          non_matching_fe_values_error.reinit(cell->dealii_iterator(),
                                              numbers::invalid_unsigned_int,
                                              numbers::invalid_unsigned_int,
                                              cell->active_fe_index());

          std::vector<types::global_dof_index> local_dof_indices(
            fe[0].dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          if (const std::optional<FEValues<dim>> &fe_values = non_matching_fe_values_error.get_inside_fe_values() )
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

    // if (params.level_set_function)
    //   {
    //     VectorType level_set;
    //     discretization.initialize_dof_vector(level_set,domain_idx);
    //     GDM::VectorTools::interpolate(mapping,
    //                                   system,
    //                                   *params.level_set_function,
    //                                   level_set);
    //     level_set.update_ghost_values();
    //     data_out.add_data_vector(level_set, "level_set");
    //   }

    VectorType analytical_solution;
    discretization.initialize_dof_vector(analytical_solution,domain_idx);
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
                 mesh_classifiers[domain_idx]->location_to_level_set(cell) !=
                   NonMatching::LocationToLevelSet::outside;
        });

    data_out.build_patches();

    std::string file_name =
      std::string("solution_") +
      ((domain_idx == 0) ? "i_" : "o_") +
      std::to_string(my_counter) + ".vtu";
    data_out.write_vtu_in_parallel(file_name);

    my_counter++;
  }

  const MPI_Comm     comm;
  ConditionalOStream pcout;

  const Parameters<dim> &params;

  Discretization<dim, Number> discretization;

  MassMatrixOperator<dim, Number>      mass_matrix_operator;
  StiffnessMatrixOperator<dim, Number> stiffness_matrix_operator;

  // std::array<TrilinosWrappers::PreconditionAMG, 2> preconditioner_amg;
  // std::array<TrilinosWrappers::PreconditionILU, 2> preconditioner_ilu;
  // std::array<TrilinosWrappers::SolverDirect, 2>    solver_direct;
  std::vector<std::shared_ptr<TrilinosWrappers::PreconditionAMG>> preconditioner_amg;
  std::vector<std::shared_ptr<TrilinosWrappers::PreconditionILU>> preconditioner_ilu;
  std::vector<std::shared_ptr<TrilinosWrappers::SolverDirect>> solver_direct;

};
