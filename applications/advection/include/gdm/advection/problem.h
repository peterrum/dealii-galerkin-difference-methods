#pragma once

#include <gdm/advection/discretization.h>
#include <gdm/advection/mass.h>
#include <gdm/advection/parameters.h>
#include <gdm/data_out.h>
#include <gdm/system.h>
#include <gdm/vector_tools.h>

using namespace dealii;

template <int dim, typename Number = double>
class AdvectionProblem
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  AdvectionProblem(const Parameters<dim> &params)
    : comm(MPI_COMM_WORLD)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0) && false)
    , params(params)
    , discretization()
    , mass_matrix_operator(discretization)
  {}

  void
  run(ConvergenceTable &table)
  {
    discretization.reinit(params);
    mass_matrix_operator.reinit(params);

    // settings
    const unsigned int fe_degree         = params.fe_degree;
    const unsigned int n_subdivisions_1D = params.n_subdivisions_1D;
    const unsigned int fe_degree_output  = 2;
    const double       dx                = (1.0 / n_subdivisions_1D);
    const double       max_vel           = 2.0;
    const double       start_t           = params.start_t;
    const double       end_t             = params.end_t;
    const double       cfl               = params.cfl;
    const double       delta_t           = dx * cfl / max_vel;
    const double       alpha             = 0.0;
    const TimeStepping::runge_kutta_method runge_kutta_method =
      TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;

    auto exact_solution     = params.exact_solution;
    auto exact_solution_der = params.exact_solution_der;
    auto advection          = params.advection;
    auto level_set_function = params.level_set_function;

    const double       ghost_parameter = 0.5;
    ConditionalOStream pcout_detail(
      std::cout, (Utilities::MPI::this_mpi_process(comm) == 0) && false);

    const auto          &mapping       = discretization.get_mapping();
    const Quadrature<1> &quadrature_1D = discretization.get_quadrature_1D();
    const Quadrature<dim - 1> &face_quadrature =
      discretization.get_face_quadrature();
    const auto                      &system = discretization.get_system();
    const AffineConstraints<Number> &constraints =
      discretization.get_affine_constraints();
    const NonMatching::MeshClassifier<dim> &mesh_classifier =
      discretization.get_mesh_classifier();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const VectorType            &level_set = discretization.get_level_set();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    const auto face_has_ghost_penalty = [&](const auto        &cell,
                                            const unsigned int face_index) {
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

    // compute mass matrix
    const auto &mass_matrix = mass_matrix_operator.get_sparse_matrix();

    // Setup solver
    this->setup_solver(mass_matrix);

    // set up initial condition
    const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      system.locally_owned_dofs(),
      system.locally_relevant_dofs(constraints),
      comm);
    BlockVectorType solution(2);
    solution.block(1).reinit(partitioner);
    GDM::VectorTools::interpolate(mapping,
                                  system,
                                  *exact_solution,
                                  solution.block(1));

    // set up BCs
    std::vector<Point<dim>> all_points;

    {
      NonMatching::RegionUpdateFlags region_update_flags;
      region_update_flags.inside = update_values | update_gradients |
                                   update_JxW_values | update_quadrature_points;
      region_update_flags.surface =
        update_values | update_gradients | update_JxW_values |
        update_quadrature_points | update_normal_vectors;

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

      for (const auto &cell : system.locally_active_cell_iterators())
        if (cell->is_locally_owned() &&
            (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
             NonMatching::LocationToLevelSet::outside))
          {
            non_matching_fe_values.reinit(cell->dealii_iterator(),
                                          numbers::invalid_unsigned_int,
                                          numbers::invalid_unsigned_int,
                                          cell->active_fe_index());

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

    solution.block(0).reinit(all_points.size());

    const auto fu_eval_bc = [&](const double time, VectorType &stage_bc) {
      exact_solution->set_time(time);
      for (unsigned int i = 0; i < all_points.size(); ++i)
        stage_bc[i] = exact_solution->value(all_points[i]);
    };

    // helper function to evaluate right-hand-side vector
    const auto fu_rhs = [&](const double           time,
                            const BlockVectorType &stage_bc_and_solution) {
      const auto &stage_bc = stage_bc_and_solution.block(0);
      const auto &solution = stage_bc_and_solution.block(1);

      BlockVectorType result;
      result.reinit(stage_bc_and_solution);

      VectorType vec_rhs;
      vec_rhs.reinit(solution); // result of assembly of rhs vector

      // evaluate derivative of bc
      exact_solution_der->set_time(time);
      for (unsigned int i = 0; i < all_points.size(); ++i)
        result.block(0)[i] = exact_solution_der->value(all_points[i]);

      // evaluate advection operator
      NonMatching::RegionUpdateFlags region_update_flags;
      region_update_flags.inside = update_values | update_gradients |
                                   update_JxW_values | update_quadrature_points;
      region_update_flags.surface =
        update_values | update_gradients | update_JxW_values |
        update_quadrature_points | update_normal_vectors;

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

      FEInterfaceValues<dim> fe_interface_values(
        mapping,
        fe,
        hp::QCollection<dim - 1>(face_quadrature),
        update_gradients | update_JxW_values | update_normal_vectors);

      advection->set_time(time);

      unsigned int point_counter = 0;

      solution.update_ghost_values();

      for (const auto &cell : system.locally_active_cell_iterators())
        if (cell->is_locally_owned() &&
            (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
             NonMatching::LocationToLevelSet::outside))
          {
            non_matching_fe_values.reinit(cell->dealii_iterator(),
                                          numbers::invalid_unsigned_int,
                                          numbers::invalid_unsigned_int,
                                          cell->active_fe_index());

            const double cell_side_length =
              cell->dealii_iterator()->minimum_vertex_distance();

            const auto &fe_values_ptr =
              non_matching_fe_values.get_inside_fe_values();

            const auto &surface_fe_values_ptr =
              non_matching_fe_values.get_surface_fe_values();

            const unsigned int n_dofs_per_cell = fe[0].dofs_per_cell;

            std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
            cell->get_dof_indices(dof_indices);

            Vector<Number> cell_vector(n_dofs_per_cell);

            // (I) cell integral
            if (fe_values_ptr)
              {
                const auto &fe_values = *fe_values_ptr;

                std::vector<Number> quadrature_values(
                  fe_values.n_quadrature_points);
                fe_values.get_function_values(solution,
                                              dof_indices,
                                              quadrature_values);

                std::vector<Tensor<1, dim, Number>> quadrature_gradients(
                  fe_values.n_quadrature_points);
                fe_values.get_function_gradients(solution,
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
                        fluxes_value[q] += quadrature_gradients[q][d] *
                                           advection->value(point, d);
                        fluxes_gradient[q][d] =
                          quadrature_values[q] * advection->value(point, d);
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

            // (II) surface integral to apply BC
            if (surface_fe_values_ptr)
              {
                const auto &fe_face_values = *surface_fe_values_ptr;

                std::vector<Number> quadrature_values(
                  fe_face_values.n_quadrature_points);
                fe_face_values.get_function_values(solution,
                                                   dof_indices,
                                                   quadrature_values);

                std::vector<Number> fluxes(fe_face_values.n_quadrature_points,
                                           0);

                for (const auto q : fe_face_values.quadrature_point_indices())
                  {
                    const auto normal = fe_face_values.normal_vector(q);
                    const auto point  = fe_face_values.quadrature_point(q);

                    for (unsigned int d = 0; d < dim; ++d)
                      fluxes[q] += normal[d] * advection->value(point, d);
                  }

                std::vector<Number> u_plus(fe_face_values.n_quadrature_points,
                                           0);

                for (const auto q : fe_face_values.quadrature_point_indices())
                  u_plus[q] = stage_bc[point_counter++];

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

            // (III) face integral to apply BC
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
                      fe_face_values.get_function_values(solution,
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
                            fluxes[q] += normal[d] * advection->value(point, d);
                        }

                      std::vector<Number> u_plus(
                        fe_face_values.n_quadrature_points, 0);

                      for (const auto q :
                           fe_face_values.quadrature_point_indices())
                        u_plus[q] = stage_bc[point_counter++];

                      for (const unsigned int q_index :
                           fe_face_values.quadrature_point_indices())
                        for (const unsigned int i :
                             fe_face_values.dof_indices())
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

            // (IV) face integral for apply GP
            for (const unsigned int f : cell->dealii_iterator()->face_indices())
              if (face_has_ghost_penalty(cell->dealii_iterator(), f))
                {
                  fe_interface_values.reinit(
                    cell->dealii_iterator(),
                    f,
                    numbers::invalid_unsigned_int,
                    cell->dealii_iterator()->neighbor(f),
                    cell->dealii_iterator()->neighbor_of_neighbor(f),
                    numbers::invalid_unsigned_int,
                    numbers::invalid_unsigned_int,
                    numbers::invalid_unsigned_int,
                    cell->active_fe_index(),
                    cell->neighbor(f)->active_fe_index());

                  const unsigned int n_interface_dofs =
                    fe_interface_values.n_current_interface_dofs();
                  Vector<double> local_stabilization(n_interface_dofs);

                  std::vector<types::global_dof_index>
                    local_interface_dof_indices;
                  cell->get_dof_indices(dof_indices);
                  for (const auto i : dof_indices)
                    local_interface_dof_indices.emplace_back(i);
                  cell->neighbor(f)->get_dof_indices(dof_indices);
                  for (const auto i : dof_indices)
                    local_interface_dof_indices.emplace_back(i);

                  std::vector<Tensor<1, dim>> jump_in_shape_gradients(
                    fe_interface_values.n_quadrature_points);

                  const FEValuesExtractors::Scalar scalar(0);

                  std::vector<double> local_dof_values(n_interface_dofs);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    local_dof_values[i] =
                      solution[local_interface_dof_indices[i]];

                  fe_interface_values[scalar]
                    .get_jump_in_function_gradients_from_local_dof_values(
                      local_dof_values, jump_in_shape_gradients);

                  for (unsigned int q = 0;
                       q < fe_interface_values.n_quadrature_points;
                       ++q)
                    {
                      const Tensor<1, dim> normal =
                        fe_interface_values.normal(q);
                      for (unsigned int i = 0; i < n_interface_dofs; ++i)
                        {
                          local_stabilization(i) -=
                            .5 * ghost_parameter * cell_side_length *
                            cell_side_length *
                            (normal *
                             fe_interface_values.jump_in_shape_gradients(i,
                                                                         q)) *
                            (normal * jump_in_shape_gradients[q]) *
                            fe_interface_values.JxW(q);
                        }
                    }

                  constraints.distribute_local_to_global(
                    local_stabilization, local_interface_dof_indices, vec_rhs);
                }

            cell->get_dof_indices(dof_indices);
            constraints.distribute_local_to_global(cell_vector,
                                                   dof_indices,
                                                   vec_rhs);
          }

      vec_rhs.compress(VectorOperation::add);

      // invert mass matrix
      this->solve(mass_matrix, result.block(1), vec_rhs);

      return result;
    };

    const auto fu_postprocessing =
      [&](const double time, const BlockVectorType &stage_bc_and_solution)
      -> std::array<double, 6> {
      static unsigned int counter = 0;

      const auto &solution = stage_bc_and_solution.block(1);

      // compute error
      exact_solution->set_time(time);

      // compute error
      const Quadrature<1> &quadrature_1D_error =
        discretization.get_quadrature_1D();

      NonMatching::RegionUpdateFlags region_update_flags_error;
      region_update_flags_error.inside =
        update_values | update_JxW_values | update_quadrature_points;
      region_update_flags_error.surface =
        update_values | update_JxW_values | update_quadrature_points;

      NonMatching::FEValues<dim> non_matching_fe_values_error(
        fe,
        quadrature_1D_error,
        region_update_flags_error,
        mesh_classifier,
        level_set_dof_handler,
        level_set);

      double local_error_Linf            = 0;
      double local_error_L1              = 0;
      double local_error_L2_squared      = 0;
      double local_error_Linf_face       = 0;
      double local_error_L1_face         = 0;
      double local_error_L2_face_squared = 0;

      solution.update_ghost_values();
      for (const auto &cell : system.locally_active_cell_iterators())
        if (cell->is_locally_owned() &&
            (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
             NonMatching::LocationToLevelSet::outside))
          {
            non_matching_fe_values_error.reinit(cell->dealii_iterator(),
                                                numbers::invalid_unsigned_int,
                                                numbers::invalid_unsigned_int,
                                                cell->active_fe_index());

            const std::optional<FEValues<dim>> &fe_values =
              non_matching_fe_values_error.get_inside_fe_values();

            const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
              &fe_surface_values =
                non_matching_fe_values_error.get_surface_fe_values();


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

                for (const unsigned int q :
                     fe_values->quadrature_point_indices())
                  {
                    const Point<dim> &point = fe_values->quadrature_point(q);
                    const double      error_at_point =
                      solution_values.at(q) - exact_solution->value(point);

                    local_error_L2_squared +=
                      Utilities::fixed_power<2>(error_at_point) *
                      fe_values->JxW(q);

                    local_error_L1 +=
                      std::abs(error_at_point) * fe_values->JxW(q);

                    local_error_Linf =
                      std::max(local_error_Linf, std::abs(error_at_point));
                  }
              }

            if (fe_surface_values)
              {
                std::vector<double> solution_values(
                  fe_surface_values->n_quadrature_points);
                fe_surface_values->get_function_values(solution,
                                                       local_dof_indices,
                                                       solution_values);

                for (const unsigned int q :
                     fe_surface_values->quadrature_point_indices())
                  {
                    const Point<dim> &point =
                      fe_surface_values->quadrature_point(q);
                    const double error_at_point =
                      solution_values.at(q) - exact_solution->value(point);

                    local_error_L2_face_squared +=
                      Utilities::fixed_power<2>(error_at_point) *
                      fe_surface_values->JxW(q);

                    local_error_L1_face +=
                      std::abs(error_at_point) * fe_surface_values->JxW(q);

                    local_error_Linf_face =
                      std::max(local_error_Linf_face, std::abs(error_at_point));
                  }
              }
          }

      const double error_Linf =
        Utilities::MPI::max(local_error_Linf, MPI_COMM_WORLD);

      const double error_L1 =
        Utilities::MPI::sum(local_error_L1, MPI_COMM_WORLD);

      const double error_L2 =
        std::sqrt(Utilities::MPI::sum(local_error_L2_squared, MPI_COMM_WORLD));

      const double error_Linf_face =
        Utilities::MPI::max(local_error_Linf_face, MPI_COMM_WORLD);

      const double error_L1_face =
        Utilities::MPI::sum(local_error_L1_face, MPI_COMM_WORLD);

      const double error_L2_face = std::sqrt(
        Utilities::MPI::sum(local_error_L2_face_squared, MPI_COMM_WORLD));

      if (pcout.is_active())
        printf("%5d %8.5f %14.8e %14.8e %14.8e\n",
               counter,
               time,
               error_L2,
               error_L1,
               error_Linf);

      // output result -> Paraview
      GDM::DataOut<dim> data_out(system, mapping, fe_degree_output);
      solution.update_ghost_values();
      data_out.add_data_vector(solution, "solution");

      VectorType level_set(partitioner);
      GDM::VectorTools::interpolate(mapping,
                                    system,
                                    *level_set_function,
                                    level_set);
      level_set.update_ghost_values();
      data_out.add_data_vector(level_set, "level_set");

      VectorType analytical_solution(partitioner);
      GDM::VectorTools::interpolate(mapping,
                                    system,
                                    *exact_solution,
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

      return {{error_Linf,
               error_L1,
               error_L2,
               error_Linf_face,
               error_L1_face,
               error_L2_face}};
    };

    // set up time stepper
    DiscreteTime time(start_t, end_t, delta_t);

    TimeStepping::ExplicitRungeKutta<BlockVectorType> rk;
    rk.initialize(runge_kutta_method);

    auto error = fu_postprocessing(0.0, solution);

    // perform time stepping
    while ((time.is_at_end() == false) && (error[2] < 1.0 /*TODO*/))
      {
        fu_eval_bc(time.get_current_time(), solution.block(0)); // evaluate bc

        rk.evolve_one_time_step(fu_rhs,
                                time.get_current_time(),
                                time.get_next_step_size(),
                                solution);

        // output result
        error =
          fu_postprocessing(time.get_current_time() + time.get_next_step_size(),
                            solution);

        time.advance_time();
      }

    table.add_value("fe_degree", fe_degree);
    table.add_value("cfl", cfl);
    table.add_value("n_subdivision", n_subdivisions_1D);
    table.add_value("error_2", error[2]);
    table.set_scientific("error_2", true);
    table.add_value("error_1", error[1]);
    table.set_scientific("error_1", true);
    table.add_value("error_inf", error[0]);
    table.set_scientific("error_inf", true);
    table.add_value("error_2_face", error[5]);
    table.set_scientific("error_2_face", true);
    table.add_value("error_1_face", error[4]);
    table.set_scientific("error_1_face", true);
    table.add_value("error_inf_face", error[3]);
    table.set_scientific("error_inf_face", true);

    pcout << std::endl;
  }

private:
  void
  setup_solver(const TrilinosWrappers::SparseMatrix &sparse_matrix,
               const unsigned int                    id = 0)
  {
    if (params.solver_name == "AMG")
      preconditioner_amg[id].initialize(sparse_matrix);
    else if (params.solver_name == "ILU")
      preconditioner_ilu[id].initialize(sparse_matrix);
    else if (params.solver_name == "direct")
      solver_direct[id].initialize(sparse_matrix);
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
          solver.solve(sparse_matrix, result, vec_rhs, preconditioner_amg[id]);
        else if (params.solver_name == "ILU")
          solver.solve(sparse_matrix, result, vec_rhs, preconditioner_ilu[id]);
        else
          AssertThrow(false, ExcNotImplemented());

        pcout << " [L] solved in " << solver_control.last_step() << std::endl;
      }
    else if (params.solver_name == "direct")
      {
        solver_direct[id].solve(sparse_matrix, result, vec_rhs);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

  const MPI_Comm     comm;
  ConditionalOStream pcout;

  const Parameters<dim> &params;

  Discretization<dim, Number> discretization;

  MassMatrixOperator<dim, Number> mass_matrix_operator;

  std::array<TrilinosWrappers::PreconditionAMG, 2> preconditioner_amg;
  std::array<TrilinosWrappers::PreconditionILU, 2> preconditioner_ilu;
  std::array<TrilinosWrappers::SolverDirect, 2>    solver_direct;
};
