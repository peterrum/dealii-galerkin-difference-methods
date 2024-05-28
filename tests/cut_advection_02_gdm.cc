// Solve cut advection problem (GDM).
//
// Setup as in:
// DoD stabilization for higher-order advection in two dimensions
// by Florian Streitbürger, Gunnar Birke, Christian Engwer, Sandra May

#include <deal.II/base/conditional_ostream.h>
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

#include <gdm/data_out.h>
#include <gdm/system.h>
#include <gdm/vector_tools.h>

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



template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_subdivisions_1D,
     const double       cfl,
     const bool         rotate,
     const double       factor = 1.0)
{
  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  // settings
  const double       phi          = (numbers::PI / 8.0) * factor; // TODO
  const double       phi_add      = rotate ? (numbers::PI / 16.0) : 0.0;
  const double       x_shift      = 0.2001;
  const unsigned int n_components = 1;
  const unsigned int fe_degree_time_stepper = fe_degree;
  const unsigned int fe_degree_level_set    = 1;
  const unsigned int fe_degree_output       = 2;
  const double       dx                     = (1.0 / n_subdivisions_1D);
  const double       max_vel                = 2.0;
  const double       sandra_factor =
    false ? (1.0 / (2 * fe_degree_time_stepper + 1)) : 1.0;
  const double delta_t = dx * cfl * sandra_factor / max_vel;
  const double start_t = 0.0;
  const double end_t   = 0.1;
  const double alpha   = 0.0;
  const TimeStepping::runge_kutta_method runge_kutta_method =
    TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;
  const std::string solver_name = "ILU";

  const MPI_Comm comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);
  ConditionalOStream pcout_detail(
    std::cout, (Utilities::MPI::this_mpi_process(comm) == 0) && false);

  ExactSolution<dim> exact_solution(x_shift, phi, phi_add);
  Functions::ConstantFunction<dim, Number> advection(
    exact_solution.get_transport_direction().begin_raw(), dim);

  // Create GDM system
  GDM::System<dim> system(comm, fe_degree, n_components);

  // Create mesh
  system.subdivided_hyper_cube(n_subdivisions_1D, 0, 1);

  // Create finite elements
  const auto &fe = system.get_fe();

  // Create mapping
  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());

  // Categorize cells
  system.categorize();

  const auto &tria = system.get_triangulation();

  // level set and classify cells
  const FE_Q<dim> fe_level_set(fe_degree_level_set);
  DoFHandler<dim> level_set_dof_handler(tria);
  level_set_dof_handler.distribute_dofs(fe_level_set);

  VectorType level_set;
  level_set.reinit(level_set_dof_handler.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(
                     level_set_dof_handler),
                   comm);

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

  level_set.update_ghost_values();
  mesh_classifier.reclassify();

  AffineConstraints<Number> constraints;
  constraints.close();

  // compute mass matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern(
    system.locally_owned_dofs(), MPI_COMM_WORLD);
  system.create_sparsity_pattern(constraints, sparsity_pattern);
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

    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : system.locally_active_cell_iterators())
      if (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
          NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values.reinit(cell->dealii_iterator(),
                                        numbers::invalid_unsigned_int,
                                        numbers::invalid_unsigned_int,
                                        cell->active_fe_index());

          const auto &fe_values = non_matching_fe_values.get_inside_fe_values();

          const unsigned int dofs_per_cell = fe[0].dofs_per_cell;

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

          // assemble
          constraints.distribute_local_to_global(cell_matrix,
                                                 dof_indices,
                                                 sparse_matrix);
        }
  }

  sparse_matrix.compress(VectorOperation::values::add);

  if (false)
    {
      for (auto &entry : sparse_matrix)
        if ((entry.row() == entry.column()))
          {
            pcout << "(" << entry.row() << ": " << entry.value() << ") "
                  << std::endl;
          }

      pcout << std::endl << std::endl;

      return;
    }

  if (false)
    {
      const double tolerance = 1e-10;

      IndexSet ghost(system.n_dofs());
      ghost.add_range(0, system.n_dofs());

      LinearAlgebra::distributed::Vector<double> diagonal(
        system.locally_owned_dofs(), ghost, comm);

      for (auto &entry : sparse_matrix)
        if ((entry.row() == entry.column()))
          {
            diagonal[entry.row()] = std::abs(entry.value());
          }

      diagonal.update_ghost_values();

      for (auto &entry : sparse_matrix)
        if ((diagonal[entry.row()] < tolerance) &&
            (diagonal[entry.column()] < tolerance))
          entry.value() = 0.0;
    }

  for (auto &entry : sparse_matrix)
    if ((entry.row() == entry.column()) && (entry.value() == 0.0))
      {
        entry.value() = 1.0;
      }

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
                                                    constraints),
                                                  comm);
  VectorType solution(partitioner);
  GDM::VectorTools::interpolate(mapping, system, exact_solution, solution);

  const auto fu_rhs = [&](const double time, const VectorType &solution) {
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

    vec_0.update_ghost_values();

    for (const auto &cell : system.locally_active_cell_iterators())
      if (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
          NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values.reinit(cell->dealii_iterator(),
                                        numbers::invalid_unsigned_int,
                                        numbers::invalid_unsigned_int,
                                        cell->active_fe_index());

          const auto &fe_values_ptr =
            non_matching_fe_values.get_inside_fe_values();

          const auto &surface_fe_values_ptr =
            non_matching_fe_values.get_surface_fe_values();

          const unsigned int n_dofs_per_cell = fe[0].dofs_per_cell;

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

          if ((phi_add != 0) && surface_fe_values_ptr)
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

          for (const auto f : cell->dealii_iterator()->face_indices())
            if (cell->dealii_iterator()->face(f)->at_boundary())
              {
                non_matching_fe_interface_values.reinit(
                  cell->dealii_iterator(),
                  f,
                  numbers::invalid_unsigned_int,
                  numbers::invalid_unsigned_int,
                  cell->active_fe_index());

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
                        const auto point  = fe_face_values.quadrature_point(q);

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
                        const auto point = fe_face_values.quadrature_point(q);
                        u_plus[q]        = exact_solution.value(point);
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

    solution.update_ghost_values();
    for (const auto &cell : system.locally_active_cell_iterators())
      if (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
          NonMatching::LocationToLevelSet::outside)
        {
          non_matching_fe_values_error.reinit(cell->dealii_iterator(),
                                              numbers::invalid_unsigned_int,
                                              numbers::invalid_unsigned_int,
                                              cell->active_fe_index());

          const std::optional<FEValues<dim>> &fe_values =
            non_matching_fe_values_error.get_inside_fe_values();


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
                    solution_values.at(q) - exact_solution.value(point);
                  error_L2_squared +=
                    Utilities::fixed_power<2>(error_at_point) *
                    fe_values->JxW(q);
                }
            }
        }

    const double error =
      std::sqrt(Utilities::MPI::sum(error_L2_squared, MPI_COMM_WORLD));

    if (pcout.is_active())
      printf("%5d %8.5f %14.8e\n", counter, time, error);

    // output result -> Paraview
    GDM::DataOut<dim> data_out(system, mapping, fe_degree_output);
    solution.update_ghost_values();
    data_out.add_data_vector(solution, "solution");

    VectorType level_set(partitioner);
    GDM::VectorTools::interpolate(mapping,
                                  system,
                                  signed_distance_sphere,
                                  level_set);
    level_set.update_ghost_values();
    data_out.add_data_vector(level_set, "level_set");

    VectorType analytical_solution(partitioner);
    GDM::VectorTools::interpolate(mapping,
                                  system,
                                  exact_solution,
                                  analytical_solution);
    analytical_solution.update_ghost_values();
    data_out.add_data_vector(analytical_solution, "analytical_solution");

    if (true)
      data_out.set_cell_selection(
        [&](const typename Triangulation<dim>::cell_iterator &cell) {
          return cell->is_active() && cell->is_locally_owned() &&
                 mesh_classifier.location_to_level_set(cell) !=
                   NonMatching::LocationToLevelSet::outside;
        });

    data_out.build_patches();

    std::string file_name = "solution_" + std::to_string(counter) + ".vtu";
    data_out.write_vtu_in_parallel(file_name);

    counter++;

    return error;
  };

  // set up time stepper
  DiscreteTime time(start_t, end_t, delta_t);

  TimeStepping::ExplicitRungeKutta<VectorType> rk;
  rk.initialize(runge_kutta_method);

  double error = fu_postprocessing(0.0);

  // perform time stepping
  while ((time.is_at_end() == false) && (error < 1.0 /*TODO*/))
    {
      rk.evolve_one_time_step(fu_rhs,
                              time.get_current_time(),
                              time.get_next_step_size(),
                              solution);

      constraints.distribute(solution);

      // output result
      error =
        fu_postprocessing(time.get_current_time() + time.get_next_step_size());

      time.advance_time();
    }

  pcout << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  if (true)
    {
      const unsigned int n_subdivisions_1D = 10;
      const double       cfl               = 0.1;

      for (const unsigned int fe_degree : {1, 3, 5})
        test<2>(fe_degree, n_subdivisions_1D, cfl, false);

      for (const unsigned int fe_degree : {1, 3, 5})
        test<2>(fe_degree, n_subdivisions_1D, cfl, true);
    }

  if (true)
    {
      const unsigned int fe_degree         = 5;
      const unsigned int n_subdivisions_1D = 10;
      const double       cfl               = 0.1;

      for (double factor = 0.5; factor <= 2.0; factor += 0.1)
        test<2>(fe_degree, n_subdivisions_1D, cfl, false, factor);
    }
}
