// Solve advection problem (FEM).

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/function.h>
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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

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



template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_subdivisions_1D,
     const double       cfl,
     const bool         weak_bc)
{
  using Number     = double;
  using VectorType = Vector<Number>;

  // settings
  const double       phi     = std::atan(0.5); // numbers::PI / 8.0; // TODO
  const double       x_shift = 0.2000;         // 0.2001
  const unsigned int fe_degree_time_stepper = fe_degree;
  const double       dx                     = (1.0 / n_subdivisions_1D);
  const double       max_vel                = 2.0;
  const double       sandra_factor =
    true ? (1.0 / (2 * fe_degree_time_stepper + 1)) : 1.0;
  const double delta_t = dx * cfl * sandra_factor / max_vel;
  const double start_t = 0.0;
  const double end_t   = 0.1;
  const double alpha   = 1.0;
  const TimeStepping::runge_kutta_method runge_kutta_method =
    TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

  AssertThrow((weak_bc == true) || (alpha == 1.0), ExcNotImplemented());

  ExactSolution<dim>                       exact_solution(x_shift, phi);
  ExactSolutionDerivative<dim>             exact_solution_der(x_shift, phi);
  Functions::ConstantFunction<dim, Number> advection(
    exact_solution.get_transport_direction().begin_raw(), dim);

  // create system
  MappingQ1<dim>  mapping;
  FE_Q<dim>       fe(fe_degree);
  QGauss<dim>     quadrature(fe_degree + 1);
  QGauss<dim - 1> face_quadrature(fe_degree + 1);

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions_1D, 0, 1, true);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Create constraints
  AffineConstraints<Number> constraints_dbc;
  if (weak_bc == false)
    {
      for (unsigned int d = 0; d < dim; ++d)
        VectorTools::interpolate_boundary_values(
          mapping, dof_handler, d * 2, exact_solution, constraints_dbc);
    }
  constraints_dbc.close();

  AffineConstraints<Number> constraints_dummy;
  constraints_dummy.close();

  // compute mass matrix
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_dummy);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<Number> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_mass_matrix<dim, dim>(mapping,
                                              dof_handler,
                                              quadrature,
                                              sparse_matrix,
                                              nullptr,
                                              constraints_dummy);

  // set up initial condition
  VectorType solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, solution);

  // set up BCs
  std::vector<Point<dim>> all_points;

  {
    FEFaceValues<dim> fe_face_values(mapping,
                                     fe,
                                     face_quadrature,
                                     update_quadrature_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              fe_face_values.reinit(cell, f);
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
        for (unsigned int d = 0; d < dim; ++d)
          VectorTools::interpolate_boundary_values(
            mapping, dof_handler, d * 2, exact_solution, constraints_dbc);
        constraints_dbc.close();

        // apply constraints
        constraints_dbc.distribute(vec_0);
      }

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);

    advection.set_time(time);

    unsigned int point_counter = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        std::vector<Number> quadrature_values(n_dofs_per_cell);
        fe_values.get_function_values(vec_0, dof_indices, quadrature_values);

        std::vector<Tensor<1, dim, Number>> quadrature_gradients(
          n_dofs_per_cell);
        fe_values.get_function_gradients(vec_0,
                                         dof_indices,
                                         quadrature_gradients);

        std::vector<Number>                 fluxes_value(n_dofs_per_cell, 0);
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

        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              fe_face_values.reinit(cell, f);

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

    // invert mass matrix
    PreconditionJacobi<SparseMatrix<Number>> preconditioner;
    preconditioner.initialize(sparse_matrix);

    ReductionControl     solver_control(100, 1.e-10, 1.e-8);
    SolverCG<VectorType> solver(solver_control);
    solver.solve(sparse_matrix, vec_2, vec_1, preconditioner);

    stage_counter++;

    return vec_2;
  };

  const auto fu_postprocessing = [&](const double time) {
    static unsigned int counter = 0;

    // compute error
    exact_solution.set_time(time);

    Vector<Number> cell_wise_error;
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      exact_solution,
                                      cell_wise_error,
                                      quadrature,
                                      VectorTools::NormType::L2_norm);
    const auto error =
      VectorTools::compute_global_error(tria,
                                        cell_wise_error,
                                        VectorTools::NormType::L2_norm);

    printf("%8.5f %14.8f\n", time, error);

    // output result -> Paraview
    dealii::DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.build_patches(mapping, fe_degree);

    std::string   file_name = "solution_" + std::to_string(counter) + ".vtu";
    std::ofstream file(file_name);
    data_out.write_vtu(file);

    counter++;
  };

  // set up time stepper
  DiscreteTime time(start_t, end_t, delta_t);

  TimeStepping::ExplicitRungeKutta<Vector<double>> rk_bc;
  rk_bc.initialize(runge_kutta_method);

  TimeStepping::ExplicitRungeKutta<VectorType> rk;
  rk.initialize(runge_kutta_method);

  fu_postprocessing(0.0);

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
      fu_postprocessing(time.get_current_time() + time.get_next_step_size());

      time.advance_time();
    }

  std::cout << std::endl;
}


int
main()
{
  if (true)
    {
      const unsigned int n_subdivisions_1D = 20;
      const double       cfl               = 0.4;

      for (const unsigned int fe_degree : {1, 3, 5})
        test<2>(fe_degree, n_subdivisions_1D, cfl, false);

      for (const unsigned int fe_degree : {1, 3, 5})
        test<2>(fe_degree, n_subdivisions_1D, cfl, true);
    }
  else
    {
      const unsigned int fe_degree         = 5;
      const unsigned int n_subdivisions_1D = 40;
      const double       cfl               = 0.4;

      test<2>(fe_degree, n_subdivisions_1D, cfl, false);
    }
}
