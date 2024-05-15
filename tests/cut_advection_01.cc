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
  ExactSolution(const double time = 0.)
    : dealii::Function<dim, Number>(1, time)
    , phi(numbers::PI / 8.0)
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
      std::cos(phi) * (position[0] - 0.2001) + std::sin(phi) * position[1];

    return std::sin(std::sqrt(2.0) * numbers::PI * x_hat / (1.0 - 0.2001));
  }

  const dealii::Tensor<1, dim> &
  get_transport_direction() const
  {
    return advection;
  }

private:
  dealii::Tensor<1, dim> advection;
  const double           phi;
};



template <int dim>
void
test(const bool do_ghost_penalty = true)
{
  using Number     = double;
  using VectorType = Vector<Number>;

  // settings
  const double       phi                    = numbers::PI / 8.0; // TODO
  const unsigned int fe_degree              = 1;
  const unsigned int fe_degree_time_stepper = 3;
  const unsigned int fe_degree_level_set    = 1;
  const unsigned int n_subdivisions_1D      = 40;
  const double       delta_t = (1.0 / n_subdivisions_1D) * 0.4 * 1.0 /
                         (2 * fe_degree_time_stepper + 1) / 2.0;
  const double                           start_t         = 0.0;
  const double                           end_t           = 0.1;
  const double                           ghost_parameter = 0.5;
  const TimeStepping::runge_kutta_method runge_kutta_method =
    TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

  ExactSolution<dim>                       exact_solution;
  Functions::ConstantFunction<dim, Number> advection(
    exact_solution.get_transport_direction().begin_raw(), dim);

  // create system
  MappingQ1<dim>        mapping;
  hp::FECollection<dim> fe;
  fe.push_back(FE_Q<dim>(fe_degree));
  QGauss<dim> quadrature(fe_degree + 1);

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

  const Point<dim> point = {0.2001, 0.0};
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
  for (unsigned int d = 0; d < dim; ++d)
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, d * 2, exact_solution, constraints);
  constraints.close();

  AffineConstraints<Number> constraints_homogeneous;
  constraints_homogeneous.close();

  // compute mass matrix
  DynamicSparsityPattern dsp(dof_handler.n_dofs());

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
                                           dsp,
                                           constraints,
                                           true,
                                           cell_coupling,
                                           face_coupling,
                                           numbers::invalid_subdomain_id,
                                           face_has_flux_coupling);
    }
  else
    {
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    }

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<Number> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  DynamicSparsityPattern dsp_homogeneous(dof_handler.n_dofs());

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
                                           dsp_homogeneous,
                                           constraints_homogeneous,
                                           true,
                                           cell_coupling,
                                           face_coupling,
                                           numbers::invalid_subdomain_id,
                                           face_has_flux_coupling);
    }
  else
    {
      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp_homogeneous,
                                      constraints_homogeneous);
    }

  SparsityPattern sparsity_pattern_homogeneous;
  sparsity_pattern_homogeneous.copy_from(dsp_homogeneous);

  SparseMatrix<Number> sparse_matrix_homogeneous;
  sparse_matrix_homogeneous.reinit(sparsity_pattern_homogeneous);

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

    hp::FEFaceValues<dim> hp_fe_face_values_m(
      fe,
      hp::QCollection<dim - 1>(QGauss<dim - 1>(fe_degree + 1)),
      update_gradients | update_JxW_values | update_normal_vectors);

    hp::FEFaceValues<dim> hp_fe_face_values_p(fe,
                                              hp::QCollection<dim - 1>(
                                                QGauss<dim - 1>(fe_degree + 1)),
                                              update_gradients);

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
                hp_fe_face_values_m.reinit(cell, f);
                hp_fe_face_values_p.reinit(cell->neighbor(f),
                                           cell->neighbor_of_neighbor(f));

                const auto &fe_face_values_m =
                  hp_fe_face_values_m.get_present_fe_values();
                const auto &fe_face_values_p =
                  hp_fe_face_values_p.get_present_fe_values();

                FullMatrix<double> local_stabilization_mm(
                  fe_face_values_m.dofs_per_cell,
                  fe_face_values_m.dofs_per_cell);
                FullMatrix<double> local_stabilization_pm(
                  fe_face_values_p.dofs_per_cell,
                  fe_face_values_m.dofs_per_cell);
                FullMatrix<double> local_stabilization_mp(
                  fe_face_values_m.dofs_per_cell,
                  fe_face_values_p.dofs_per_cell);
                FullMatrix<double> local_stabilization_pp(
                  fe_face_values_p.dofs_per_cell,
                  fe_face_values_p.dofs_per_cell);

                for (const unsigned int q :
                     fe_face_values_m.quadrature_point_indices())
                  {
                    const Tensor<1, dim> normal =
                      fe_face_values_m.normal_vector(q);

                    for (const auto i : fe_face_values_m.dof_indices())
                      for (const auto j : fe_face_values_m.dof_indices())
                        {
                          local_stabilization_mm(i, j) +=
                            .5 * ghost_parameter * cell_side_length *
                            (normal * fe_face_values_m.shape_grad(i, q)) *
                            (normal * fe_face_values_m.shape_grad(j, q)) *
                            fe_face_values_m.JxW(q);
                        }

                    for (const auto i : fe_face_values_p.dof_indices())
                      for (const auto j : fe_face_values_m.dof_indices())
                        {
                          local_stabilization_pm(i, j) -=
                            .5 * ghost_parameter * cell_side_length *
                            (normal * fe_face_values_p.shape_grad(i, q)) *
                            (normal * fe_face_values_m.shape_grad(j, q)) *
                            fe_face_values_m.JxW(q);
                        }

                    for (const auto i : fe_face_values_m.dof_indices())
                      for (const auto j : fe_face_values_p.dof_indices())
                        {
                          local_stabilization_mp(i, j) -=
                            .5 * ghost_parameter * cell_side_length *
                            (normal * fe_face_values_m.shape_grad(i, q)) *
                            (normal * fe_face_values_p.shape_grad(j, q)) *
                            fe_face_values_m.JxW(q);
                        }

                    for (const auto i : fe_face_values_m.dof_indices())
                      for (const auto j : fe_face_values_m.dof_indices())
                        {
                          local_stabilization_pp(i, j) +=
                            .5 * ghost_parameter * cell_side_length *
                            (normal * fe_face_values_p.shape_grad(i, q)) *
                            (normal * fe_face_values_p.shape_grad(j, q)) *
                            fe_face_values_m.JxW(q);
                        }
                  }

                std::vector<types::global_dof_index> local_dof_indices_m(
                  fe_face_values_m.dofs_per_cell);
                std::vector<types::global_dof_index> local_dof_indices_p(
                  fe_face_values_p.dofs_per_cell);

                cell->get_dof_indices(local_dof_indices_m);
                cell->neighbor(f)->get_dof_indices(local_dof_indices_p);

                constraints.distribute_local_to_global(local_stabilization_mm,
                                                       local_dof_indices_m,
                                                       local_dof_indices_m,
                                                       sparse_matrix);
                constraints.distribute_local_to_global(local_stabilization_pm,
                                                       local_dof_indices_p,
                                                       local_dof_indices_m,
                                                       sparse_matrix);
                constraints.distribute_local_to_global(local_stabilization_mp,
                                                       local_dof_indices_m,
                                                       local_dof_indices_p,
                                                       sparse_matrix);
                constraints.distribute_local_to_global(local_stabilization_pp,
                                                       local_dof_indices_p,
                                                       local_dof_indices_p,
                                                       sparse_matrix);

                constraints_homogeneous.distribute_local_to_global(
                  local_stabilization_mm,
                  local_dof_indices_m,
                  local_dof_indices_m,
                  sparse_matrix_homogeneous);
                constraints_homogeneous.distribute_local_to_global(
                  local_stabilization_pm,
                  local_dof_indices_p,
                  local_dof_indices_m,
                  sparse_matrix_homogeneous);
                constraints_homogeneous.distribute_local_to_global(
                  local_stabilization_mp,
                  local_dof_indices_m,
                  local_dof_indices_p,
                  sparse_matrix_homogeneous);
                constraints_homogeneous.distribute_local_to_global(
                  local_stabilization_pp,
                  local_dof_indices_p,
                  local_dof_indices_p,
                  sparse_matrix_homogeneous);
              }

          // assemble
          constraints.distribute_local_to_global(cell_matrix,
                                                 dof_indices,
                                                 sparse_matrix);

          constraints_homogeneous.distribute_local_to_global(
            cell_matrix, dof_indices, sparse_matrix_homogeneous);
        }
  }

  for (auto &entry : sparse_matrix)
    if ((entry.row() == entry.column()) && (entry.value() == 0.0))
      {
        entry.value() = 1.0;
      }

  for (auto &entry : sparse_matrix_homogeneous)
    if ((entry.row() == entry.column()) && (entry.value() == 0.0))
      {
        entry.value() = 1.0;
      }

  // set up initial condition
  VectorType solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, solution);

  const auto fu_rhs = [&](const double time, const VectorType &solution) {
    VectorType vec_0, vec_1, vec_2;
    vec_0.reinit(solution); // for applying constraints
    vec_1.reinit(solution); // result of assembly of rhs vector
    vec_2.reinit(solution); // result of inversion mass matrix

    vec_0 = solution;

    // update constraints
    exact_solution.set_time(time);

    constraints.clear();
    for (unsigned int d = 0; d < dim; ++d)
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, d * 2, exact_solution, constraints);
    constraints.close();

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

    advection.set_time(time);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (mesh_classifier.location_to_level_set(cell) !=
          NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values.reinit(cell);

          const auto &fe_values = non_matching_fe_values.get_inside_fe_values();

          const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();

          std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          Vector<Number> cell_vector(n_dofs_per_cell);

          if (fe_values)
            {
              std::vector<Tensor<1, dim, Number>> quadrature_gradients(
                fe_values->n_quadrature_points);
              fe_values->get_function_gradients(vec_0,
                                                dof_indices,
                                                quadrature_gradients);

              std::vector<Number> fluxes(fe_values->n_quadrature_points, 0);

              for (const auto q : fe_values->quadrature_point_indices())
                {
                  const auto point = fe_values->quadrature_point(q);

                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      fluxes[q] +=
                        quadrature_gradients[q][d] * advection.value(point, d);
                    }
                }

              for (const unsigned int q_index :
                   fe_values->quadrature_point_indices())
                for (const unsigned int i : fe_values->dof_indices())
                  cell_vector(i) -= fluxes[q_index] *
                                    fe_values->shape_value(i, q_index) *
                                    fe_values->JxW(q_index);
            }

          constraints.distribute_local_to_global(cell_vector,
                                                 dof_indices,
                                                 vec_1);
        }

    VectorType vec_dbc, vec_dbc_in;
    vec_dbc.reinit(solution);
    vec_dbc_in.reinit(solution);

    constraints.distribute(vec_dbc_in);
    sparse_matrix_homogeneous.vmult(vec_dbc, vec_dbc_in);
    constraints.set_zero(vec_dbc);

    vec_1 -= vec_dbc;

    // invert mass matrix
    PreconditionJacobi<SparseMatrix<Number>> preconditioner;
    preconditioner.initialize(sparse_matrix);

    ReductionControl     solver_control(1000, 1.e-10, 1.e-8);
    SolverCG<VectorType> solver(solver_control);
    solver.solve(sparse_matrix, vec_2, vec_1, preconditioner);

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

    std::cout << time << " " << error << std::endl;

    // output result -> Paraview
    dealii::DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

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
