#pragma once

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <gdm/wave/discretization.h>

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
    , nitsche_parameter(-1.0)
  {}

  void
  reinit(const Parameters<dim> &params)
  {
    this->ghost_parameter_A = params.ghost_parameter_A;
    this->nitsche_parameter = params.nitsche_parameter;

    this->function_interface_dbc = params.function_interface_dbc;
    this->function_rhs           = params.function_rhs;
  }

  const TrilinosWrappers::SparseMatrix &
  get_sparse_matrix() const
  {
    compute_sparse_matrix();

    return sparse_matrix;
  }

  void
  compute_rhs_internal(VectorType                           &vec_rhs,
                       const VectorType                     &solution,
                       const bool                            compute_impl_part,
                       const double                          time,
                       const NonMatching::LocationToLevelSet location) const
  {
    const NonMatching::LocationToLevelSet inverse_location =
      (location == NonMatching::LocationToLevelSet::inside) ?
        NonMatching::LocationToLevelSet::outside :
        NonMatching::LocationToLevelSet::inside;

    // 0) extract information from discretization class
    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const Quadrature<1> &quadrature_1D = discretization.get_quadrature_1D();
    const Quadrature<dim - 1> &face_quadrature =
      discretization.get_face_quadrature();
    const GDM::System<dim>          &system = discretization.get_system();
    const AffineConstraints<Number> &constraints =
      discretization.get_affine_constraints();
    const NonMatching::MeshClassifier<dim> &mesh_classifier =
      discretization.get_mesh_classifier();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const VectorType            &level_set = discretization.get_level_set();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    AssertThrow(ghost_parameter_A != -1.0, ExcNotImplemented());

    if (function_interface_dbc)
      function_interface_dbc->set_time(time);

    if (function_rhs)
      function_rhs->set_time(time);

    const auto face_has_ghost_penalty = [&](const auto        &cell,
                                            const unsigned int face_index) {
      if (cell->at_boundary(face_index))
        return false;

      const NonMatching::LocationToLevelSet cell_location =
        mesh_classifier.location_to_level_set(cell);

      const NonMatching::LocationToLevelSet neighbor_location =
        mesh_classifier.location_to_level_set(cell->neighbor(face_index));

      if (cell_location == NonMatching::LocationToLevelSet::intersected &&
          neighbor_location != inverse_location)
        return true;

      if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
          cell_location != inverse_location)
        return true;

      return false;
    };

    NonMatching::RegionUpdateFlags region_update_flags;
    if (location == NonMatching::LocationToLevelSet::inside)
      region_update_flags.inside = update_values | update_gradients |
                                   update_JxW_values | update_quadrature_points;
    else if (location == NonMatching::LocationToLevelSet::outside)
      region_update_flags.outside = update_values | update_gradients |
                                    update_JxW_values |
                                    update_quadrature_points;
    else
      AssertThrow(false, ExcNotImplemented());
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

    solution.update_ghost_values();

    for (const auto &cell : system.locally_active_cell_iterators())
      if (cell->is_locally_owned() &&
          (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
           inverse_location))
        {
          non_matching_fe_values.reinit(cell->dealii_iterator(),
                                        numbers::invalid_unsigned_int,
                                        numbers::invalid_unsigned_int,
                                        cell->active_fe_index());

          const double cell_side_length =
            cell->dealii_iterator()->minimum_vertex_distance();

          const unsigned int n_dofs_per_cell = fe[0].dofs_per_cell;

          std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          Vector<Number> cell_vector(n_dofs_per_cell);

          // (I) cell integral
          if (const auto &fe_values_ptr =
                (location == NonMatching::LocationToLevelSet::inside) ?
                  non_matching_fe_values.get_inside_fe_values() :
                  non_matching_fe_values.get_outside_fe_values())
            {
              const auto &fe_values = *fe_values_ptr;

              std::vector<Tensor<1, dim, Number>> quadrature_gradients(
                fe_values.n_quadrature_points);
              fe_values.get_function_gradients(solution,
                                               dof_indices,
                                               quadrature_gradients);

              for (const unsigned int q : fe_values.quadrature_point_indices())
                {
                  const Point<dim> &point = fe_values.quadrature_point(q);
                  for (const unsigned int i : fe_values.dof_indices())
                    {
                      // left hand side: (∇v, ∇u)
                      if (compute_impl_part)
                        cell_vector(i) -= fe_values.shape_grad(i, q) *
                                          quadrature_gradients[q] *
                                          fe_values.JxW(q);

                      // right hand side: (v, f)
                      if (function_rhs)
                        cell_vector(i) += function_rhs->value(point) *
                                          fe_values.shape_value(i, q) *
                                          fe_values.JxW(q);
                    }
                }
            }

          // (II) surface integral to apply BC
          if (function_interface_dbc)
            if (const auto &surface_fe_values_ptr =
                  non_matching_fe_values.get_surface_fe_values())
              {
                const auto &surface_fe_values = *surface_fe_values_ptr;

                std::vector<Number> quadrature_values(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_values(solution,
                                                      dof_indices,
                                                      quadrature_values);

                std::vector<Tensor<1, dim, Number>> quadrature_gradients(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_gradients(solution,
                                                         dof_indices,
                                                         quadrature_gradients);


                for (const unsigned int q :
                     surface_fe_values.quadrature_point_indices())
                  {
                    const Point<dim> &point =
                      surface_fe_values.quadrature_point(q);
                    const Tensor<1, dim> &normal =
                      surface_fe_values.normal_vector(q);
                    for (const unsigned int i : surface_fe_values.dof_indices())
                      {
                        // left hand side:
                        // - <v, ∂u/∂n> - <∂v/∂n, u> + γ_D/h <v, u>
                        if (compute_impl_part)
                          cell_vector(i) -=
                            (-normal * surface_fe_values.shape_grad(i, q) *
                               quadrature_values[q] +
                             -normal * quadrature_gradients[q] *
                               surface_fe_values.shape_value(i, q) +
                             nitsche_parameter / cell_side_length *
                               surface_fe_values.shape_value(i, q) *
                               quadrature_values[q]) *
                            surface_fe_values.JxW(q);

                        // right hand side: <γ_D/h v - ∂v/∂n, g_D>
                        cell_vector(i) +=
                          function_interface_dbc->value(point) *
                          (nitsche_parameter / cell_side_length *
                             surface_fe_values.shape_value(i, q) -
                           normal * surface_fe_values.shape_grad(i, q)) *
                          surface_fe_values.JxW(q);
                      }
                  }
              }

          // (IV) face integral for apply GP
          if (compute_impl_part)
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
                          // γ_A j(v, u) / h^2 with j(v, u)= ∑ h^3 <∂v/∂n,
                          // ∂u/∂n>
                          local_stabilization(i) -=
                            .5 * ghost_parameter_A * cell_side_length *
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
  }

  void
  compute_rhs(VectorType       &vec_rhs,
              const VectorType &solution,
              const bool        compute_impl_part,
              const double      time) const
  {
    compute_rhs_internal(vec_rhs,
                         solution,
                         compute_impl_part,
                         time,
                         NonMatching::LocationToLevelSet::inside);
  }

  void
  compute_rhs(BlockVectorType       &vec_rhs,
              const BlockVectorType &solution,
              const bool             compute_impl_part,
              const double           time) const
  {
    compute_rhs_internal(vec_rhs.block(0),
                         solution.block(0),
                         compute_impl_part,
                         time,
                         NonMatching::LocationToLevelSet::inside);
    compute_rhs_internal(vec_rhs.block(1),
                         solution.block(1),
                         compute_impl_part,
                         time,
                         NonMatching::LocationToLevelSet::outside);
  }

private:
  const Discretization<dim, Number> &discretization;

  double ghost_parameter_A;
  double nitsche_parameter;

  std::shared_ptr<Function<dim>> function_interface_dbc;
  std::shared_ptr<Function<dim>> function_rhs;

  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;

  void
  compute_sparse_matrix() const
  {
    // 0) extract information from discretization class
    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const Quadrature<1> &quadrature_1D = discretization.get_quadrature_1D();
    const Quadrature<dim - 1> &face_quadrature =
      discretization.get_face_quadrature();
    const GDM::System<dim>          &system = discretization.get_system();
    const AffineConstraints<Number> &constraints =
      discretization.get_affine_constraints();
    const NonMatching::MeshClassifier<dim> &mesh_classifier =
      discretization.get_mesh_classifier();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const VectorType            &level_set = discretization.get_level_set();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    AssertThrow(ghost_parameter_A != -1.0, ExcNotImplemented());
    AssertThrow(nitsche_parameter != -1.0, ExcNotImplemented());

    // 1) create sparsity pattern
    if (sparse_matrix.m() == 0 || sparse_matrix.n() == 0)
      {
        sparsity_pattern.reinit(system.locally_owned_dofs(), MPI_COMM_WORLD);
        system.create_flux_sparsity_pattern(constraints, sparsity_pattern);
        sparsity_pattern.compress();

        sparse_matrix.reinit(sparsity_pattern);
      }
    else
      {
        sparse_matrix = 0.0;
      }

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
      mapping,
      fe,
      hp::QCollection<dim - 1>(face_quadrature),
      update_gradients | update_JxW_values | update_normal_vectors);

    std::vector<types::global_dof_index> dof_indices;
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

          const unsigned int dofs_per_cell = fe[0].dofs_per_cell;

          // compute element stiffness matrix
          FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);

          // (I) cell integral
          if (const auto &fe_values =
                non_matching_fe_values.get_inside_fe_values())
            {
              for (const unsigned int q_index :
                   fe_values->quadrature_point_indices())
                {
                  for (const unsigned int i : fe_values->dof_indices())
                    for (const unsigned int j : fe_values->dof_indices())
                      // (∇v, ∇u)
                      cell_matrix(i, j) += fe_values->shape_grad(i, q_index) *
                                           fe_values->shape_grad(j, q_index) *
                                           fe_values->JxW(q_index);
                }
            }

          // (II) surface integral to apply BC
          if (function_interface_dbc)
            if (const auto &surface_fe_values_ptr =
                  non_matching_fe_values.get_surface_fe_values())
              {
                const auto &surface_fe_values = *surface_fe_values_ptr;
                for (const unsigned int q :
                     surface_fe_values.quadrature_point_indices())
                  {
                    const Tensor<1, dim> &normal =
                      surface_fe_values.normal_vector(q);
                    for (const unsigned int i : surface_fe_values.dof_indices())
                      for (const unsigned int j :
                           surface_fe_values.dof_indices())
                        {
                          // - <v, ∂u/∂n> - <∂v/∂n, u> + γ_D/h <v, u>
                          cell_matrix(i, j) +=
                            (-normal * surface_fe_values.shape_grad(i, q) *
                               surface_fe_values.shape_value(j, q) +
                             -normal * surface_fe_values.shape_grad(j, q) *
                               surface_fe_values.shape_value(i, q) +
                             nitsche_parameter / cell_side_length *
                               surface_fe_values.shape_value(i, q) *
                               surface_fe_values.shape_value(j, q)) *
                            surface_fe_values.JxW(q);
                        }
                  }
              }

          // (II) face integral to apply GP
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
                          // clang-format off
                          // γ_A j(v, u) / h^2 with j(v, u)= ∑ h^3 <∂v/∂n, ∂u/∂n>
                          local_stabilization(i, j) +=
                            .5 * ghost_parameter_A * cell_side_length *
                            cell_side_length * cell_side_length *
                            (normal * fe_interface_values.jump_in_shape_gradients(i, q)) *
                            (normal * fe_interface_values.jump_in_shape_gradients(j, q)) *
                            fe_interface_values.JxW(q);
                          // clang-format on
                        }
                  }

                std::vector<types::global_dof_index>
                  local_interface_dof_indices;
                dof_indices.resize(dofs_per_cell);
                cell->get_dof_indices(dof_indices);
                for (const auto i : dof_indices)
                  local_interface_dof_indices.emplace_back(i);
                cell->neighbor(f)->get_dof_indices(dof_indices);
                for (const auto i : dof_indices)
                  local_interface_dof_indices.emplace_back(i);

                sparse_matrix.add(local_interface_dof_indices,
                                  local_stabilization);
              }

          // get indices
          dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          // assemble
          constraints.distribute_local_to_global(cell_matrix,
                                                 dof_indices,
                                                 sparse_matrix);
        }

    sparse_matrix.compress(VectorOperation::values::add);

    for (auto &entry : sparse_matrix)
      if ((entry.row() == entry.column()) && (entry.value() == 0.0))
        entry.value() = 1.0;
  }
};
