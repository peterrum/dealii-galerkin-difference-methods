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
  ExactSolution(const double time = 0.)
    : dealii::Function<dim, Number>(1, time)
    , wave_number(2.)
  {
    advection[0] = 1.0;
    if (dim > 1)
      advection[1] = 0.15;
    if (dim > 2)
      advection[2] = -0.05;
  }

  virtual double
  value(const dealii::Point<dim> &p, const unsigned int = 1) const override
  {
    double                       t        = this->get_time();
    const dealii::Tensor<1, dim> position = p - t * advection;
    double result = std::sin(wave_number * position[0] * dealii::numbers::PI);
    for (unsigned int d = 1; d < dim; ++d)
      result *= std::cos(wave_number * position[d] * dealii::numbers::PI);
    return result;
  }

  const dealii::Tensor<1, dim> &
  get_transport_direction() const
  {
    return advection;
  }

private:
  dealii::Tensor<1, dim> advection;
  const double           wave_number;
};



template <int dim>
void
test()
{
  using Number     = double;
  using VectorType = Vector<Number>;

  // settings
  const unsigned int fe_degree         = 3;
  const unsigned int n_subdivisions_1D = 40;
  const double       delta_t           = 1.0 / n_subdivisions_1D * 0.2;
  const double       start_t           = 0.0;
  const double       end_t             = 0.1;
  const double       alpha             = 0.0;
  const TimeStepping::runge_kutta_method runge_kutta_method =
    TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

  ExactSolution<dim>                       exact_solution;
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

  AffineConstraints<Number> constraints;
  constraints.close();

  // compute mass matrix
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<Number> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_mass_matrix<dim, dim>(
    mapping, dof_handler, quadrature, sparse_matrix, nullptr, constraints);

  // set up initial condition
  VectorType solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, solution);

  const auto fu_rhs = [&](const double time, const VectorType &solution) {
    VectorType vec_0, vec_1, vec_2;
    vec_0.reinit(solution); // for applying constraints
    vec_1.reinit(solution); // result of assembly of rhs vector
    vec_2.reinit(solution); // result of inversion mass matrix

    vec_0 = solution;

    // apply constraints
    constraints.distribute(vec_0);

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
                  const auto point = fe_face_values.quadrature_point(q);
                  u_plus[q]        = exact_solution.value(point);
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

        constraints.distribute_local_to_global(cell_vector, dof_indices, vec_1);
      }

    // invert mass matrix
    PreconditionJacobi<SparseMatrix<Number>> preconditioner;
    preconditioner.initialize(sparse_matrix);

    ReductionControl     solver_control(100, 1.e-10, 1.e-8);
    SolverCG<VectorType> solver(solver_control);
    solver.solve(sparse_matrix, vec_2, vec_1, preconditioner);

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

    std::cout << time << " " << error << std::endl;

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

  TimeStepping::ExplicitRungeKutta<VectorType> rk;
  rk.initialize(runge_kutta_method);

  fu_postprocessing(0.0);

  // perform time stepping
  while (time.is_at_end() == false)
    {
      rk.evolve_one_time_step(fu_rhs,
                              time.get_current_time(),
                              time.get_next_step_size(),
                              solution);
      time.advance_time();

      constraints.distribute(solution);

      // output result
      fu_postprocessing(time.get_current_time() + time.get_next_step_size());
    }
}


int
main()
{
  test<2>();
}
