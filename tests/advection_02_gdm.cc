// Solve advection problem (GDM).

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/time_stepping.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <gdm/data_out.h>
#include <gdm/matrix_creator.h>
#include <gdm/system.h>
#include <gdm/vector_tools.h>

#include <fstream>

using namespace dealii;

template <int dim, typename Number = double>
class ExactSolution : public dealii::Function<dim, Number>
{
public:
  ExactSolution(const double x_shift, const double phi, const double time = 0.)
    : dealii::Function<dim, Number>(1, time)
    , x_shift(x_shift)
    , phi(phi)
  {
    advection[0] = 2.0 * std::cos(phi);
    advection[1] = 2.0 * std::sin(phi);
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
                          const double time = 0.)
    : dealii::Function<dim, Number>(1, time)
    , x_shift(x_shift)
    , phi(phi)
  {
    advection[0] = 2.0 * std::cos(phi);
    advection[1] = 2.0 * std::sin(phi);
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



template <typename SparseMatrixType>
void
eigenvalue_estimates(const SparseMatrixType &matrix_a)
{
  LAPACKFullMatrix<double> lapack_full_matrix(matrix_a.m(), matrix_a.n());


  for (const auto &entry : matrix_a)
    lapack_full_matrix.set(entry.row(), entry.column(), entry.value());

  lapack_full_matrix.compute_eigenvalues();

  std::vector<double> eigenvalues;

  for (unsigned int i = 0; i < lapack_full_matrix.m(); ++i)
    eigenvalues.push_back(lapack_full_matrix.eigenvalue(i).real());

  std::sort(eigenvalues.begin(), eigenvalues.end());

  printf("%10.2e %10.2e %10.2e \n",
         eigenvalues[0],
         eigenvalues.back(),
         eigenvalues.back() / eigenvalues[0]);
}


template <int dim>
void
test(ConvergenceTable  &table,
     const unsigned int fe_degree,
     const unsigned int n_subdivisions_1D,
     const double       cfl,
     const bool         weak_bc)
{
  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  // settings
  const double       phi     = std::atan(0.5); // numbers::PI / 8.0; // TODO
  const double       x_shift = 0.2000;         // 0.2001
  const unsigned int n_components           = 1;
  const unsigned int fe_degree_time_stepper = 1;
  const unsigned int fe_degree_output       = 2;
  const double       dx                     = (1.0 / n_subdivisions_1D);
  const double       max_vel                = 2.0;
  const double       sandra_factor =
    false ? (1.0 / (2 * fe_degree_time_stepper + 1)) : 1.0;
  const double delta_t = dx * cfl * sandra_factor / max_vel;
  const double start_t = 0.0;
  const double end_t   = 0.10;
  const double alpha   = 1.0;
  const TimeStepping::runge_kutta_method runge_kutta_method =
    TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;
  const std::string solver_name = "ILU";

  const MPI_Comm comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                           (Utilities::MPI::this_mpi_process(comm) == 0) &&
                             false);
  ConditionalOStream pcout_detail(
    std::cout, (Utilities::MPI::this_mpi_process(comm) == 0) && false);

  AssertThrow((weak_bc == true) || (alpha == 1.0), ExcNotImplemented());

  ExactSolution<dim>                       exact_solution(x_shift, phi);
  ExactSolutionDerivative<dim>             exact_solution_der(x_shift, phi);
  Functions::ConstantFunction<dim, Number> advection(
    exact_solution.get_transport_direction().begin_raw(), dim);

  // Create GDM system
  GDM::System<dim> system(comm, fe_degree, n_components);

  // Create mesh
  system.subdivided_hyper_cube(n_subdivisions_1D);

  // Create finite elements
  const auto &fe = system.get_fe();

  // Create mapping
  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());

  // Create quadrature
  hp::QCollection<dim> quadrature;
  quadrature.push_back(QGauss<dim>(fe_degree + 1));

  hp::QCollection<dim - 1> face_quadrature;
  face_quadrature.push_back(QGauss<dim - 1>(fe_degree + 1));

  // Create constraints
  AffineConstraints<Number> constraints_dbc(system.locally_active_dofs());
  if (weak_bc == false)
    {
      for (unsigned int d = 0; d < dim; ++d)
        system.interpolate_boundary_values(mapping,
                                           d * 2,
                                           exact_solution,
                                           constraints_dbc);
    }
  constraints_dbc.close();

  AffineConstraints<Number> constraints_dummy(system.locally_active_dofs());
  constraints_dummy.close();

  // Categorize cells
  system.categorize();

  // compute mass matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern(
    system.locally_owned_dofs(), MPI_COMM_WORLD);
  system.create_sparsity_pattern(constraints_dummy, sparsity_pattern);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);
  GDM::MatrixCreator::create_mass_matrix(
    mapping, system, quadrature, sparse_matrix, constraints_dummy);

  TrilinosWrappers::PreconditionAMG preconditioner_amg;
  TrilinosWrappers::PreconditionILU preconditioner_ilu;
  TrilinosWrappers::SolverDirect    solver_direct;

  if (solver_name == "AMG")
    preconditioner_amg.initialize(sparse_matrix);
  else if (solver_name == "ILU")
    preconditioner_ilu.initialize(sparse_matrix);
  else if (solver_name == "direct")
    solver_direct.initialize(sparse_matrix);
  else
    AssertThrow(false, ExcNotImplemented());

  // set up initial condition
  const auto partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(system.locally_owned_dofs(),
                                                  system.locally_relevant_dofs(
                                                    constraints_dummy),
                                                  comm);
  VectorType solution(partitioner);
  GDM::VectorTools::interpolate(mapping, system, exact_solution, solution);

  // set up BCs
  std::vector<Point<dim>> all_points;

  {
    hp::FEFaceValues<dim> fe_face_values_collection(mapping,
                                                    fe,
                                                    face_quadrature,
                                                    update_values |
                                                      update_gradients |
                                                      update_quadrature_points |
                                                      update_JxW_values |
                                                      update_normal_vectors);
    for (const auto &cell : system.locally_active_cell_iterators())
      {
        for (const auto f : cell->dealii_iterator()->face_indices())
          if (weak_bc && cell->dealii_iterator()->face(f)->at_boundary())
            {
              fe_face_values_collection.reinit(cell->dealii_iterator(),
                                               f,
                                               numbers::invalid_unsigned_int,
                                               numbers::invalid_unsigned_int,
                                               cell->active_fe_index());

              const auto &fe_face_values =
                fe_face_values_collection.get_present_fe_values();

              for (const auto q : fe_face_values.quadrature_point_indices())
                {
                  all_points.emplace_back(fe_face_values.quadrature_point(q));
                }
            }
      }
  }

  std::vector<Vector<double>> stage_bcs;

  const auto fu_eval_bc = [&](const double time) {
    exact_solution.set_time(time);

    Vector<double> stage_bc(all_points.size());

    for (unsigned int i = 0; i < all_points.size(); ++i)
      stage_bc[i] = exact_solution.value(all_points[i]);

    return stage_bc;
  };

  const auto fu_bc = [&](const double time, const Vector<double> &solution) {
    if (false)
      {
        exact_solution.set_time(time);

        Vector<double> stage_bc(all_points.size());

        for (unsigned int i = 0; i < all_points.size(); ++i)
          stage_bc[i] = exact_solution.value(all_points[i]);
        stage_bcs.emplace_back(stage_bc);

        return solution;
      }
    else
      {
        exact_solution_der.set_time(time);

        stage_bcs.emplace_back(solution);

        Vector<double> stage_bc(all_points.size());

        for (unsigned int i = 0; i < all_points.size(); ++i)
          stage_bc[i] = exact_solution_der.value(all_points[i]);

        return stage_bc;
      }
  };

  // helper function to evaluate right-hand-side vector
  unsigned int stage_counter = 0;
  const auto   fu_rhs = [&](const double time, const VectorType &solution) {
    VectorType vec_0, vec_1, vec_2;
    vec_0.reinit(solution); // for applying constraints
    vec_1.reinit(solution); // result of assembly of rhs vector
    vec_2.reinit(solution); // result of inversion mass matrix

    vec_0 = solution;

    exact_solution.set_time(time);

    // apply constraints
    if (weak_bc == false)
      {
        constraints_dbc.clear();
        constraints_dbc.reinit(system.locally_active_dofs());
        for (unsigned int d = 0; d < dim; ++d)
          system.interpolate_boundary_values(mapping,
                                             d * 2,
                                             exact_solution,
                                             constraints_dbc);
        constraints_dbc.close();

        // apply constraints
        constraints_dbc.distribute(vec_0);
      }

    hp::FEValues<dim> fe_values_collection(mapping,
                                           fe,
                                           quadrature,
                                           update_gradients | update_values |
                                             update_JxW_values |
                                             update_quadrature_points);

    hp::FEFaceValues<dim> fe_face_values_collection(mapping,
                                                    fe,
                                                    face_quadrature,
                                                    update_values |
                                                      update_gradients |
                                                      update_quadrature_points |
                                                      update_JxW_values |
                                                      update_normal_vectors);

    advection.set_time(time);

    unsigned int point_counter = 0;

    vec_0.update_ghost_values();

    for (const auto &cell : system.locally_active_cell_iterators())
      {
        fe_values_collection.reinit(cell->dealii_iterator(),
                                    numbers::invalid_unsigned_int,
                                    numbers::invalid_unsigned_int,
                                    cell->active_fe_index());

        const auto &fe_values = fe_values_collection.get_present_fe_values();

        const unsigned int n_dofs_per_cell =
          fe_values.get_fe().n_dofs_per_cell();

        std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        std::vector<Number> quadrature_values(n_dofs_per_cell);
        fe_values.get_function_values(vec_0, dof_indices, quadrature_values);

        std::vector<Tensor<1, dim, Number>> quadrature_gradients(
          n_dofs_per_cell);
        fe_values.get_function_gradients(vec_0,
                                         dof_indices,
                                         quadrature_gradients);

        std::vector<Number>                 fluxes_value(n_dofs_per_cell);
        std::vector<Tensor<1, dim, Number>> fluxes_gradient(n_dofs_per_cell);

        for (const auto q : fe_values.quadrature_point_indices())
          {
            const auto point = fe_values.quadrature_point(q);

            for (unsigned int d = 0; d < dim; ++d)
              {
                fluxes_value[q] +=
                  quadrature_gradients[q][d] * advection.value(point, d);
                fluxes_gradient[q][d] =
                  quadrature_values[q] * advection.value(point, d);
              }
          }

        Vector<Number> cell_vector(n_dofs_per_cell);
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            cell_vector(i) += alpha * (-fluxes_value[q_index] *
                                       fe_values.shape_value(i, q_index) *
                                       fe_values.JxW(q_index)) +
                              (1 - alpha) * (fluxes_gradient[q_index] *
                                             fe_values.shape_grad(i, q_index) *
                                             fe_values.JxW(q_index));

        for (const auto f : cell->dealii_iterator()->face_indices())
          if (weak_bc && cell->dealii_iterator()->face(f)->at_boundary())
            {
              fe_face_values_collection.reinit(cell->dealii_iterator(),
                                               f,
                                               numbers::invalid_unsigned_int,
                                               numbers::invalid_unsigned_int,
                                               cell->active_fe_index());

              const auto &fe_face_values =
                fe_face_values_collection.get_present_fe_values();

              std::vector<Number> quadrature_values(
                fe_face_values.n_quadrature_points);
              fe_face_values.get_function_values(vec_0,
                                                 dof_indices,
                                                 quadrature_values);

              std::vector<Number> fluxes(n_dofs_per_cell, 0);

              for (const auto q : fe_face_values.quadrature_point_indices())
                {
                  const auto normal = fe_face_values.normal_vector(q);
                  const auto point  = fe_face_values.quadrature_point(q);

                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      fluxes[q] += normal[d] * advection.value(point, d);
                    }
                }

              std::vector<Number> u_plus(n_dofs_per_cell, 0);

              for (const auto q : fe_face_values.quadrature_point_indices())
                {
                  u_plus[q] = stage_bcs[stage_counter][point_counter++];
                }

              for (const unsigned int q_index :
                   fe_face_values.quadrature_point_indices())
                for (const unsigned int i : fe_face_values.dof_indices())
                  cell_vector(i) +=
                    fluxes[q_index] *
                    (alpha * quadrature_values[q_index] -
                     ((fluxes[q_index] >= 0.0) ? quadrature_values[q_index] :
                                                   u_plus[q_index])) *
                    fe_face_values.shape_value(i, q_index) *
                    fe_face_values.JxW(q_index);
            }

        constraints_dummy.distribute_local_to_global(cell_vector,
                                                     dof_indices,
                                                     vec_1);
      }

    vec_1.compress(VectorOperation::add);

    // invert mass matrix
    if (solver_name == "AMG" || solver_name == "ILU")
      {
        ReductionControl     solver_control(1000, 1.e-20, 1.e-14);
        SolverCG<VectorType> solver(solver_control);

        if (solver_name == "AMG")
          solver.solve(sparse_matrix, vec_2, vec_1, preconditioner_amg);
        else if (solver_name == "ILU")
          solver.solve(sparse_matrix, vec_2, vec_1, preconditioner_ilu);
        else
          AssertThrow(false, ExcNotImplemented());

        pcout_detail << " [L] solved in " << solver_control.last_step()
                     << std::endl;
      }
    else if (solver_name == "direct")
      {
        solver_direct.solve(sparse_matrix, vec_2, vec_1);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }

    constraints_dbc.set_zero(vec_2);

    stage_counter++;

    return vec_2;
  };

  // helper function for postprocessing
  const auto fu_postprocessing = [&](const double time) {
    static unsigned int counter = 0;

    // compute error
    exact_solution.set_time(time);

    Vector<Number> cell_wise_error;
    solution.update_ghost_values();
    GDM::VectorTools::integrate_difference(mapping,
                                           system,
                                           solution,
                                           exact_solution,
                                           cell_wise_error,
                                           quadrature,
                                           VectorTools::NormType::L2_norm);
    const auto error =
      VectorTools::compute_global_error(system.get_triangulation(),
                                        cell_wise_error,
                                        VectorTools::NormType::L2_norm);

    if (pcout.is_active())
      printf("%5d %8.5f %14.8e\n", counter, time, error);

    // output result -> Paraview
    GDM::DataOut<dim> data_out(system, mapping, fe_degree_output);
    data_out.add_data_vector(solution, "solution");

    VectorType analytical_solution(partitioner);
    GDM::VectorTools::interpolate(mapping,
                                  system,
                                  exact_solution,
                                  analytical_solution);
    analytical_solution.update_ghost_values();
    data_out.add_data_vector(analytical_solution, "analytical_solution");
    data_out.build_patches();

    data_out.write_vtu_in_parallel("solution_" + std::to_string(counter) +
                                   ".vtu");

    counter++;

    return error;
  };

  // set up time stepper
  DiscreteTime time(start_t, end_t, delta_t);

  TimeStepping::ExplicitRungeKutta<Vector<double>> rk_bc;
  rk_bc.initialize(runge_kutta_method);

  TimeStepping::ExplicitRungeKutta<VectorType> rk;
  rk.initialize(runge_kutta_method);

  double error = fu_postprocessing(0.0);

  // perform time stepping
  while (time.is_at_end() == false)
    {
      stage_bcs.clear();
      Vector<double> solution_bc = fu_eval_bc(time.get_current_time());
      rk_bc.evolve_one_time_step(fu_bc,
                                 time.get_current_time(),
                                 time.get_next_step_size(),
                                 solution_bc);

      stage_counter = 0;
      rk.evolve_one_time_step(fu_rhs,
                              time.get_current_time(),
                              time.get_next_step_size(),
                              solution);

      constraints_dbc.distribute(solution);

      // output result
      error =
        fu_postprocessing(time.get_current_time() + time.get_next_step_size());

      time.advance_time();
    }

  table.add_value("fe_degree", fe_degree);
  table.add_value("cfl", cfl);
  table.add_value("n_subdivision", n_subdivisions_1D);
  table.add_value("error", error);
  table.set_scientific("error", true);

  if (false)
    {
      eigenvalue_estimates(sparse_matrix);
    }

  pcout << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  ConvergenceTable table;

  if (false)
    {
      const unsigned int n_subdivisions_1D = 10;
      const double       cfl               = 0.1;

      for (const unsigned int fe_degree : {1, 3, 5})
        test<2>(table, fe_degree, n_subdivisions_1D, cfl, false);

      for (const unsigned int fe_degree : {1, 3, 5})
        test<2>(table, fe_degree, n_subdivisions_1D, cfl, true);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          table.write_text(std::cout);
          std::cout << std::endl;
        }
    }

  if (false)
    {
      const unsigned int fe_degree         = 5;
      const unsigned int n_subdivisions_1D = 40;
      const double       cfl               = 0.4;

      test<2>(table, fe_degree, n_subdivisions_1D, cfl, false);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          table.write_text(std::cout);
          std::cout << std::endl;
        }
    }

  if (true)
    {
      for (const unsigned int fe_degree : {3, 5})
        {
          for (const double cfl : {0.4, 0.2, 0.1, 0.05, 0.025})
            {
              for (unsigned int n_subdivisions_1D = 10;
                   n_subdivisions_1D <= 100;
                   n_subdivisions_1D += 10)
                test<2>(table, fe_degree, n_subdivisions_1D, cfl, true);

              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                {
                  table.write_text(std::cout);
                  std::cout << std::endl;
                }

              table.clear();
            }
        }
    }
}
