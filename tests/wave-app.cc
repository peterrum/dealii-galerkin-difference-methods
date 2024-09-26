#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/time_stepping.h>

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

  WaveProblem()
    : comm(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
  {}

  void
  run()
  {
    Discretization<dim, Number> discretization;

    MassMatrixOperator<dim, Number>      mass_matrix_operator(discretization);
    StiffnessMatrixOperator<dim, Number> stiffness_matrix_operator(
      discretization);

    discretization.reinit();

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
        const double                           start_t = 0.0; // TODO
        const double                           end_t   = 0.1; // TODO
        const double                           delta_t = 0.1; // TODO
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
        AssertThrow(false, ExcNotImplemented());
      }
    else if (simulation_type == "wave-rk")
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

private:
  void
  setup_solver(const TrilinosWrappers::SparseMatrix &sparse_matrix)
  {
    std::string solver_name = "AMG"; // TODO

    if (solver_name == "AMG")
      preconditioner_amg.initialize(sparse_matrix);
    else if (solver_name == "ILU")
      preconditioner_ilu.initialize(sparse_matrix);
    else if (solver_name == "direct")
      solver_direct.initialize(sparse_matrix);
    else
      AssertThrow(false, ExcNotImplemented());
  }

  void
  solve(const TrilinosWrappers::SparseMatrix &sparse_matrix,
        VectorType                           &result,
        const VectorType                     &vec_rhs)
  {
    std::string solver_name = "AMG"; // TODO

    if (solver_name == "AMG" || solver_name == "ILU")
      {
        ReductionControl     solver_control(1000, 1.e-20, 1.e-14);
        SolverCG<VectorType> solver(solver_control);

        if (solver_name == "AMG")
          solver.solve(sparse_matrix, result, vec_rhs, preconditioner_amg);
        else if (solver_name == "ILU")
          solver.solve(sparse_matrix, result, vec_rhs, preconditioner_ilu);
        else
          AssertThrow(false, ExcNotImplemented());

        pcout << " [L] solved in " << solver_control.last_step() << std::endl;
      }
    else if (solver_name == "direct")
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
    WaveProblem<1>().run();
  else if (dim == 2)
    WaveProblem<2>().run();
  else
    AssertThrow(false, ExcNotImplemented());
}
