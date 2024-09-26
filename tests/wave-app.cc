#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/time_stepping.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include "wave-discretization.h"
#include "wave-mass.h"
#include "wave-stiffness.h"


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
  {}

  void
  run()
  {
    Discretization<dim, Number> discretization;

    MassMatrixOperator<dim, Number>      mass_matrix_operator(discretization);
    StiffnessMatrixOperator<dim, Number> stiffness_matrix_operator(
      discretization);

    discretization.reinit(params);
    mass_matrix_operator.reinit(params);
    stiffness_matrix_operator.reinit(params);

    const std::string simulation_type = "poisson";

    if (simulation_type == "poisson")
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
    else if (simulation_type == "heat-rk")
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
    else if (simulation_type == "heat-impl")
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
    else if (simulation_type == "wave-rk")
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
  }

private:
  void
  setup_solver(const TrilinosWrappers::SparseMatrix &sparse_matrix)
  {
    if (params.solver_name == "AMG")
      preconditioner_amg.initialize(sparse_matrix);
    else if (params.solver_name == "ILU")
      preconditioner_ilu.initialize(sparse_matrix);
    else if (params.solver_name == "direct")
      solver_direct.initialize(sparse_matrix);
    else
      AssertThrow(false, ExcNotImplemented());
  }

  void
  solve(const TrilinosWrappers::SparseMatrix &sparse_matrix,
        VectorType                           &result,
        const VectorType                     &vec_rhs)
  {
    if (params.solver_name == "AMG" || params.solver_name == "ILU")
      {
        ReductionControl solver_control(params.solver_max_iterations,
                                        params.solver_abs_tolerance,
                                        params.solver_rel_tolerance);

        SolverCG<VectorType> solver(solver_control);

        if (params.solver_name == "AMG")
          solver.solve(sparse_matrix, result, vec_rhs, preconditioner_amg);
        else if (params.solver_name == "ILU")
          solver.solve(sparse_matrix, result, vec_rhs, preconditioner_ilu);
        else
          AssertThrow(false, ExcNotImplemented());

        pcout << " [L] solved in " << solver_control.last_step() << std::endl;
      }
    else if (params.solver_name == "direct")
      {
        solver_direct.solve(sparse_matrix, result, vec_rhs);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

  void
  postprocess(const double time, const VectorType &vec)
  {
    (void)vec;
    (void)time;
  }

  const MPI_Comm     comm;
  ConditionalOStream pcout;

  const Parameters<dim> &params;

  TrilinosWrappers::PreconditionAMG preconditioner_amg;
  TrilinosWrappers::PreconditionILU preconditioner_ilu;
  TrilinosWrappers::SolverDirect    solver_direct;
};

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim = 2;

  if (dim == 1)
    {
      Parameters<1> params;
      WaveProblem<1>(params).run();
    }
  else if (dim == 2)
    {
      Parameters<2> params;
      WaveProblem<2>(params).run();
    }
  else
    AssertThrow(false, ExcNotImplemented());
}
