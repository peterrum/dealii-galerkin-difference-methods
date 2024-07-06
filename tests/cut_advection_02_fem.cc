// Solve cut advection problem (FEM).
//
// Setup as in:
// DoD stabilization for higher-order advection in two dimensions
// by Florian Streitbürger, Gunnar Birke, Christian Engwer, Sandra May

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
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

#include <fstream>

using namespace dealii;

template <int dim, typename Number = double>
class ExactSolution : public dealii::Function<dim, Number>
{
public:
  ExactSolution(const double x_shift,
                const double phi,
                const double phi_add,
                const double time = 0.)
    : dealii::Function<dim, Number>(1, time)
    , x_shift(x_shift)
    , phi(phi)
  {
    advection[0] = 2.0 * std::cos(phi + phi_add);
    advection[1] = 2.0 * std::sin(phi + phi_add);
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
                          const double phi_add,
                          const double time = 0.)
    : dealii::Function<dim, Number>(1, time)
    , x_shift(x_shift)
    , phi(phi)
  {
    advection[0] = 2.0 * std::cos(phi + phi_add);
    advection[1] = 2.0 * std::sin(phi + phi_add);
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
     const bool         do_ghost_penalty = false)
{
  using Number     = double;
  using VectorType = Vector<Number>;

  // settings
  const double       phi     = std::atan(0.5); // numbers::PI / 8.0; // TODO
  const double       phi_add = numbers::PI / 16.0;
  const double       x_shift = 0.2000; // 0.2001
  const unsigned int fe_degree_time_stepper = fe_degree;
  const unsigned int fe_degree_level_set    = 1;
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

  const double ghost_parameter = 0.5;

  ExactSolution<dim>           exact_solution(x_shift, phi, phi_add);
  ExactSolutionDerivative<dim> exact_solution_der(x_shift, phi, phi_add);
  Functions::ConstantFunction<dim, Number> advection(
    exact_solution.get_transport_direction().begin_raw(), dim);

  // create system
  MappingQ1<dim>        mapping;
  hp::FECollection<dim> fe;
  fe.push_back(FE_Q<dim>(fe_degree));
  QGauss<dim>     quadrature(fe_degree + 1);
  QGauss<dim - 1> face_quadrature(fe_degree + 1);

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions_1D, 0, 1, true);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // level set and classify cells
  const FE_Q<dim> fe_level_set(fe_degree_level_set);
  DoFHandler<dim> level_set_dof_handler(tria);
  level_set_dof_handler.distribute_dofs(fe_level_set);

  Vector<double> level_set;
  level_set.reinit(level_set_dof_handler.n_dofs());

  NonMatching::MeshClassifier<dim> mesh_classifier(level_set_dof_handler,
                                                   level_set);

  const Point<dim> point = {x_shift, 0.0};
  Tensor<1, dim>   normal;
  normal[0] = +std::sin(phi);
  normal[1] = -std::cos(phi);
  const Functions::SignedDistance::Plane<dim> signed_distance_sphere(point,
                                                                     normal);
  VectorTools::interpolate(level_set_dof_handler,
                           signed_distance_sphere,
                           level_set);

  mesh_classifier.reclassify();

  const auto face_has_ghost_penalty = [&](const auto        &cell,
                                          const unsigned int face_index) {
    if (!do_ghost_penalty)
      return false;

    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;

    return false;
  };

  AffineConstraints<Number> constraints;
  constraints.close();

  // compute mass matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern(
    dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

  if (do_ghost_penalty)
    {
      const auto face_has_flux_coupling = [&](const auto        &cell,
                                              const unsigned int face_index) {
        return face_has_ghost_penalty(cell, face_index);
      };

      Table<2, DoFTools::Coupling> cell_coupling(1, 1);
      Table<2, DoFTools::Coupling> face_coupling(1, 1);
      cell_coupling[0][0] = DoFTools::always;
      face_coupling[0][0] = DoFTools::always;

      DoFTools::make_flux_sparsity_pattern(dof_handler,
                                           sparsity_pattern,
                                           constraints,
                                           true,
                                           cell_coupling,
                                           face_coupling,
                                           numbers::invalid_subdomain_id,
                                           face_has_flux_coupling);
    }
  else
    {
      DoFTools::make_sparsity_pattern(dof_handler,
                                      sparsity_pattern,
                                      constraints);
    }
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  {
    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    FEInterfaceValues<dim> fe_interface_values(
      fe,
      hp::QCollection<dim - 1>(QGauss<dim - 1>(fe_degree + 1)),
      update_gradients | update_JxW_values | update_normal_vectors);

    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (mesh_classifier.location_to_level_set(cell) !=
          NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values.reinit(cell);

          const double cell_side_length = cell->minimum_vertex_distance();

          const auto &fe_values = non_matching_fe_values.get_inside_fe_values();

          const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

          // get indices
          dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          // compute element stiffness matrix
          FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);

          if (fe_values)
            {
              for (const unsigned int q_index :
                   fe_values->quadrature_point_indices())
                {
                  for (const unsigned int i : fe_values->dof_indices())
                    for (const unsigned int j : fe_values->dof_indices())
                      cell_matrix(i, j) += fe_values->shape_value(i, q_index) *
                                           fe_values->shape_value(j, q_index) *
                                           fe_values->JxW(q_index);
                }
            }

          for (const unsigned int f : cell->face_indices())
            if (face_has_ghost_penalty(cell, f))
              {
                fe_interface_values.reinit(cell,
                                           f,
                                           numbers::invalid_unsigned_int,
                                           cell->neighbor(f),
                                           cell->neighbor_of_neighbor(f),
                                           numbers::invalid_unsigned_int);

                const unsigned int n_interface_dofs =
                  fe_interface_values.n_current_interface_dofs();
                FullMatrix<double> local_stabilization(n_interface_dofs,
                                                       n_interface_dofs);
                for (unsigned int q = 0;
                     q < fe_interface_values.n_quadrature_points;
                     ++q)
                  {
                    const Tensor<1, dim> normal = fe_interface_values.normal(q);
                    for (unsigned int i = 0; i < n_interface_dofs; ++i)
                      for (unsigned int j = 0; j < n_interface_dofs; ++j)
                        {
                          local_stabilization(i, j) +=
                            .5 * ghost_parameter * cell_side_length *
                            cell_side_length * cell_side_length *
                            (normal *
                             fe_interface_values.jump_in_shape_gradients(i,
                                                                         q)) *
                            (normal *
                             fe_interface_values.jump_in_shape_gradients(j,
                                                                         q)) *
                            fe_interface_values.JxW(q);
                        }
                  }

                const std::vector<types::global_dof_index>
                  local_interface_dof_indices =
                    fe_interface_values.get_interface_dof_indices();

                local_stabilization.print(std::cout);
                std::cout << std::endl;

                sparse_matrix.add(local_interface_dof_indices,
                                  local_stabilization);
              }

          // assemble
          constraints.distribute_local_to_global(cell_matrix,
                                                 dof_indices,
                                                 sparse_matrix);
        }
  }

  sparse_matrix.compress(VectorOperation::values::add);

  for (auto &entry : sparse_matrix)
    if ((entry.row() == entry.column()) && (entry.value() == 0.0))
      {
        entry.value() = 1.0;
      }

  TrilinosWrappers::PreconditionAMG preconditioner_ilu;
  preconditioner_ilu.initialize(sparse_matrix);

  // set up initial condition
  VectorType solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, solution);

  // set up BCs
  std::vector<Point<dim>> all_points;

  {
    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    NonMatching::RegionUpdateFlags region_update_flags_face;
    region_update_flags_face.inside =
      update_values | update_gradients | update_JxW_values |
      update_quadrature_points | update_normal_vectors;

    NonMatching::FEInterfaceValues<dim> non_matching_fe_interface_values(
      fe,
      quadrature_1D,
      region_update_flags_face,
      mesh_classifier,
      level_set_dof_handler,
      level_set);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (mesh_classifier.location_to_level_set(cell) !=
          NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values.reinit(cell);

          const auto &surface_fe_values_ptr =
            non_matching_fe_values.get_surface_fe_values();

          if (surface_fe_values_ptr)
            {
              const auto &fe_face_values = *surface_fe_values_ptr;

              for (const auto q : fe_face_values.quadrature_point_indices())
                {
                  all_points.emplace_back(fe_face_values.quadrature_point(q));
                }
            }

          for (const auto f : cell->face_indices())
            if (cell->face(f)->at_boundary())
              {
                non_matching_fe_interface_values.reinit(cell, f);

                const auto &fe_interface_values =
                  non_matching_fe_interface_values.get_inside_fe_values();

                if (fe_interface_values)
                  {
                    const auto &fe_face_values =
                      fe_interface_values->get_fe_face_values(0);

                    for (const auto q :
                         fe_face_values.quadrature_point_indices())
                      {
                        all_points.emplace_back(
                          fe_face_values.quadrature_point(q));
                      }
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
    constraints.distribute(vec_0);

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    NonMatching::RegionUpdateFlags region_update_flags_face;
    region_update_flags_face.inside =
      update_values | update_gradients | update_JxW_values |
      update_quadrature_points | update_normal_vectors;

    NonMatching::FEInterfaceValues<dim> non_matching_fe_interface_values(
      fe,
      quadrature_1D,
      region_update_flags_face,
      mesh_classifier,
      level_set_dof_handler,
      level_set);

    advection.set_time(time);

    unsigned int point_counter = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (mesh_classifier.location_to_level_set(cell) !=
          NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values.reinit(cell);

          const auto &fe_values_ptr =
            non_matching_fe_values.get_inside_fe_values();

          const auto &surface_fe_values_ptr =
            non_matching_fe_values.get_surface_fe_values();

          const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();

          std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          Vector<Number> cell_vector(n_dofs_per_cell);

          if (fe_values_ptr)
            {
              const auto &fe_values = *fe_values_ptr;

              std::vector<Number> quadrature_values(
                fe_values.n_quadrature_points);
              fe_values.get_function_values(vec_0,
                                            dof_indices,
                                            quadrature_values);

              std::vector<Tensor<1, dim, Number>> quadrature_gradients(
                fe_values.n_quadrature_points);
              fe_values.get_function_gradients(vec_0,
                                               dof_indices,
                                               quadrature_gradients);

              std::vector<Number> fluxes_value(fe_values.n_quadrature_points,
                                               0);
              std::vector<Tensor<1, dim, Number>> fluxes_gradient(
                fe_values.n_quadrature_points);

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

              for (const unsigned int q_index :
                   fe_values.quadrature_point_indices())
                for (const unsigned int i : fe_values.dof_indices())
                  cell_vector(i) +=
                    alpha * (-fluxes_value[q_index] *
                             fe_values.shape_value(i, q_index) *
                             fe_values.JxW(q_index)) +
                    (1 - alpha) * (fluxes_gradient[q_index] *
                                   fe_values.shape_grad(i, q_index) *
                                   fe_values.JxW(q_index));
            }

          if (surface_fe_values_ptr)
            {
              const auto &fe_face_values = *surface_fe_values_ptr;

              std::vector<Number> quadrature_values(
                fe_face_values.n_quadrature_points);
              fe_face_values.get_function_values(vec_0,
                                                 dof_indices,
                                                 quadrature_values);

              std::vector<Number> fluxes(fe_face_values.n_quadrature_points, 0);

              for (const auto q : fe_face_values.quadrature_point_indices())
                {
                  const auto normal = fe_face_values.normal_vector(q);
                  const auto point  = fe_face_values.quadrature_point(q);

                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      fluxes[q] += normal[d] * advection.value(point, d);
                    }
                }

              std::vector<Number> u_plus(fe_face_values.n_quadrature_points, 0);

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

          for (const auto f : cell->face_indices())
            if (cell->face(f)->at_boundary())
              {
                non_matching_fe_interface_values.reinit(cell, f);

                const auto &fe_interface_values =
                  non_matching_fe_interface_values.get_inside_fe_values();

                if (fe_interface_values)
                  {
                    const auto &fe_face_values =
                      fe_interface_values->get_fe_face_values(0);

                    std::vector<Number> quadrature_values(
                      fe_face_values.n_quadrature_points);
                    fe_face_values.get_function_values(vec_0,
                                                       dof_indices,
                                                       quadrature_values);

                    std::vector<Number> fluxes(
                      fe_face_values.n_quadrature_points, 0);

                    for (const auto q :
                         fe_face_values.quadrature_point_indices())
                      {
                        const auto normal = fe_face_values.normal_vector(q);
                        const auto point = fe_face_values.quadrature_point(q);

                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            fluxes[q] += normal[d] * advection.value(point, d);
                          }
                      }

                    std::vector<Number> u_plus(
                      fe_face_values.n_quadrature_points, 0);

                    for (const auto q :
                         fe_face_values.quadrature_point_indices())
                      {
                        u_plus[q] = stage_bcs[stage_counter][point_counter++];
                      }

                    for (const unsigned int q_index :
                         fe_face_values.quadrature_point_indices())
                      for (const unsigned int i : fe_face_values.dof_indices())
                        cell_vector(i) +=
                          fluxes[q_index] *
                          (alpha * quadrature_values[q_index] -
                           ((fluxes[q_index] >= 0.0) ?
                                quadrature_values[q_index] :
                                u_plus[q_index])) *
                          fe_face_values.shape_value(i, q_index) *
                          fe_face_values.JxW(q_index);
                  }
              }

          constraints.distribute_local_to_global(cell_vector,
                                                 dof_indices,
                                                 vec_1);
        }

    // invert mass matrix
    ReductionControl     solver_control(10000, 1.e-10, 1.e-8);
    SolverCG<VectorType> solver(solver_control);
    solver.solve(sparse_matrix, vec_2, vec_1, preconditioner_ilu);

    stage_counter++;

    return vec_2;
  };

  const auto fu_postprocessing = [&](const double time) {
    static unsigned int counter = 0;

    // compute error
    exact_solution.set_time(time);

    // compute error
    const QGauss<1> quadrature_1D_error(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags_error;
    region_update_flags_error.inside =
      update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values_error(
      fe,
      quadrature_1D_error,
      region_update_flags_error,
      mesh_classifier,
      level_set_dof_handler,
      level_set);

    double error_L2_squared = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (mesh_classifier.location_to_level_set(cell) !=
          NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values_error.reinit(cell);

          const std::optional<FEValues<dim>> &fe_values =
            non_matching_fe_values_error.get_inside_fe_values();

          if (fe_values)
            {
              std::vector<double> solution_values(
                fe_values->n_quadrature_points);
              fe_values->get_function_values(solution, solution_values);

              for (const unsigned int q : fe_values->quadrature_point_indices())
                {
                  const Point<dim> &point = fe_values->quadrature_point(q);
                  const double      error_at_point =
                    solution_values.at(q) - exact_solution.value(point);
                  error_L2_squared +=
                    Utilities::fixed_power<2>(error_at_point) *
                    fe_values->JxW(q);
                }
            }
        }

    const double error = std::sqrt(error_L2_squared);

    printf("%8.5f %14.8f\n", time, error);

    // output result -> Paraview
    dealii::DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    VectorType analytical_solution(dof_handler.n_dofs());
    VectorTools::interpolate(dof_handler, exact_solution, analytical_solution);
    data_out.add_data_vector(dof_handler,
                             analytical_solution,
                             "analytical_solution");

    data_out.set_cell_selection(
      [&](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
               mesh_classifier.location_to_level_set(cell) !=
                 NonMatching::LocationToLevelSet::outside;
      });

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

      constraints.distribute(solution);

      // output result
      fu_postprocessing(time.get_current_time() + time.get_next_step_size());

      time.advance_time();
    }

  std::cout << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  if (true)
    {
      const unsigned int n_subdivisions_1D = 20;
      const double       cfl               = 0.4;

      for (const unsigned int fe_degree : {3})
        test<2>(fe_degree, n_subdivisions_1D, cfl);
    }
  else
    {
      const unsigned int fe_degree         = 5;
      const unsigned int n_subdivisions_1D = 40;
      const double       cfl               = 0.4;

      test<2>(fe_degree, n_subdivisions_1D, cfl);
    }
}
