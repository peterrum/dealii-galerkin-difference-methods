// Solve advection problem (GDM).

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/time_stepping.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>

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



template <int dim>
void
test(const bool use_mass_lumping)
{
  using Number     = double;
  using VectorType = Vector<Number>;

  // settings
  const double       phi                    = numbers::PI / 8.0; // TODO
  const double       x_shift                = 0.2000;            // 0.2001
  const unsigned int n_components           = 1;
  const unsigned int fe_degree              = 3;
  const unsigned int fe_degree_time_stepper = fe_degree;
  const unsigned int n_subdivisions_1D      = 40;
  const unsigned int fe_degree_output       = 2;
  const double       delta_t = (1.0 / n_subdivisions_1D) * 0.4 * 1.0 /
                         (2 * fe_degree_time_stepper + 1) / 2.0;
  const double                           start_t = 0.0;
  const double                           end_t   = 0.01;
  const double                           alpha   = 1.0;
  const TimeStepping::runge_kutta_method runge_kutta_method =
    TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

  ExactSolution<dim>                       exact_solution(x_shift, phi);
  Functions::ConstantFunction<dim, Number> advection(
    exact_solution.get_transport_direction().begin_raw(), dim);

  // Create GDM system
  GDM::System<dim> system(fe_degree, n_components);

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
  AffineConstraints<Number> constraints;
  constraints.close();

  // Categorize cells
  system.categorize();

  // compute mass matrix
  DynamicSparsityPattern dsp(system.n_dofs());
  system.create_sparsity_pattern(constraints, dsp);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<Number>       sparse_matrix;
  DiagonalMatrix<VectorType> diagonal_matrix;

  if (use_mass_lumping == false)
    {
      sparse_matrix.reinit(sparsity_pattern);
      GDM::MatrixCreator::create_mass_matrix(
        mapping, system, quadrature, sparse_matrix, constraints);
    }
  else
    {
      diagonal_matrix.get_vector().reinit(system.n_dofs());
      GDM::MatrixCreator::create_lumped_mass_matrix(
        mapping, system, quadrature, diagonal_matrix.get_vector(), constraints);
    }


  // set up initial condition
  VectorType solution(system.n_dofs());
  GDM::VectorTools::interpolate(mapping, system, exact_solution, solution);

  // helper function to evaluate right-hand-side vector
  const auto fu_rhs = [&](const double time, const VectorType &solution) {
    VectorType vec_0, vec_1, vec_2;
    vec_0.reinit(solution); // for applying constraints
    vec_1.reinit(solution); // result of assembly of rhs vector
    vec_2.reinit(solution); // result of inversion mass matrix

    vec_0 = solution;

    // apply constraints
    constraints.distribute(vec_0);

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
          if (cell->dealii_iterator()->face(f)->at_boundary())
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

    if (use_mass_lumping == false)
      {
        // invert mass matrix
        PreconditionJacobi<SparseMatrix<Number>> preconditioner;
        preconditioner.initialize(sparse_matrix);

        ReductionControl     solver_control(100, 1.e-10, 1.e-8);
        SolverCG<VectorType> solver(solver_control);
        solver.solve(sparse_matrix, vec_2, vec_1, preconditioner);
      }
    else
      {
        diagonal_matrix.vmult(vec_2, vec_1);
      }

    return vec_2;
  };

  // helper function for postprocessing
  const auto fu_postprocessing = [&](const double time) {
    static unsigned int counter = 0;

    // compute error
    exact_solution.set_time(time);

    Vector<Number> cell_wise_error;
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

    std::cout << time << " " << error << std::endl;

    // output result -> Paraview
    GDM::DataOut<dim> data_out(system, mapping, fe_degree_output);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    std::ofstream file("solution_" + std::to_string(counter) + ".vtu");
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
  test<2>(/*use_mass_lumping=*/false);
  test<2>(/*use_mass_lumping=*/true);
}
