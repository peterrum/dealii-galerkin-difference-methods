#pragma once

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <gdm/advection/discretization.h>

using namespace dealii;

template <unsigned int dim, typename Number>
class StiffnessMatrixOperator
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  StiffnessMatrixOperator(const Discretization<dim, Number> &discretization)
    : discretization(discretization)
    , ghost_parameter_A(-1.0)
  {}

  void
  reinit(const Parameters<dim> &params)
  {
    this->ghost_parameter_A  = params.ghost_parameter_A;
    this->exact_solution     = params.exact_solution;
    this->exact_solution_der = params.exact_solution_der;
    this->advection          = params.advection;

    const Quadrature<1> &quadrature_1D = discretization.get_quadrature_1D();
    const Quadrature<dim - 1> &face_quadrature =
      discretization.get_face_quadrature();
    const auto &system = discretization.get_system();
    const NonMatching::MeshClassifier<dim> &mesh_classifier =
      discretization.get_mesh_classifier();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const VectorType            &level_set = discretization.get_level_set();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

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
  }

  void
  initialize_dof_vector(BlockVectorType &vec) const
  {
    vec.reinit(2);
    vec.block(0).reinit(all_points.size());
    discretization.initialize_dof_vector(vec.block(1));
  }

  void
  initialize_time_step(BlockVectorType &stage_bc_and_solution,
                       const double     time)
  {
    const auto fu_eval_bc = [&](const double time, VectorType &stage_bc) {
      exact_solution->set_time(time);
      for (unsigned int i = 0; i < all_points.size(); ++i)
        stage_bc[i] = exact_solution->value(all_points[i]);
    };

    fu_eval_bc(time, stage_bc_and_solution.block(0));
  }

  void
  compute_rhs(BlockVectorType       &vec_rhs,
              const BlockVectorType &stage_bc_and_solution,
              const double           time) const
  {
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

    const double alpha = 0.0;

    const auto &stage_bc = stage_bc_and_solution.block(0);
    const auto &solution = stage_bc_and_solution.block(1);


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


    // evaluate derivative of bc
    exact_solution_der->set_time(time);
    for (unsigned int i = 0; i < all_points.size(); ++i)
      vec_rhs.block(0)[i] = exact_solution_der->value(all_points[i]);

    // evaluate advection operator
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
                      fluxes_value[q] +=
                        quadrature_gradients[q][d] * advection->value(point, d);
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

              std::vector<Number> fluxes(fe_face_values.n_quadrature_points, 0);

              for (const auto q : fe_face_values.quadrature_point_indices())
                {
                  const auto normal = fe_face_values.normal_vector(q);
                  const auto point  = fe_face_values.quadrature_point(q);

                  for (unsigned int d = 0; d < dim; ++d)
                    fluxes[q] += normal[d] * advection->value(point, d);
                }

              std::vector<Number> u_plus(fe_face_values.n_quadrature_points, 0);

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
                        const auto point  = fe_face_values.quadrature_point(q);

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
                    const Tensor<1, dim> normal = fe_interface_values.normal(q);
                    for (unsigned int i = 0; i < n_interface_dofs; ++i)
                      {
                        local_stabilization(i) -=
                          .5 * ghost_parameter_A * cell_side_length *
                          cell_side_length *
                          (normal *
                           fe_interface_values.jump_in_shape_gradients(i, q)) *
                          (normal * jump_in_shape_gradients[q]) *
                          fe_interface_values.JxW(q);
                      }
                  }

                constraints.distribute_local_to_global(
                  local_stabilization,
                  local_interface_dof_indices,
                  vec_rhs.block(1));
              }

          cell->get_dof_indices(dof_indices);
          constraints.distribute_local_to_global(cell_vector,
                                                 dof_indices,
                                                 vec_rhs.block(1));
        }

    vec_rhs.block(1).compress(VectorOperation::add);
  }

private:
  const Discretization<dim, Number> &discretization;

  double ghost_parameter_A;

  std::shared_ptr<Function<dim>> exact_solution;
  std::shared_ptr<Function<dim>> exact_solution_der;
  std::shared_ptr<Function<dim>> advection;

  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;

  std::vector<Point<dim>> all_points;
};
