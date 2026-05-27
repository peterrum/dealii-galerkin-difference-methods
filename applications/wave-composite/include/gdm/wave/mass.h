#pragma once

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <gdm/wave/discretization.h>

using namespace dealii;

template <unsigned int dim, typename Number>
class MassMatrixOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  MassMatrixOperator(const Discretization<dim, Number>    &discretization)
    : discretization(discretization)
    , ghost_parameter_M(-1.0)
  {}

  void
  reinit(const Parameters<dim> &params)
  {
    this->ghost_parameter_M = params.ghost_parameter_M;
  }

  const std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &
  get_sparse_matrix() const
  {
      block_sparse_matrix.clear();
      block_sparse_matrix.resize(discretization.get_level_sets().size());
      for (size_t i = 0; i < discretization.get_level_sets().size(); ++i)
      {
          block_sparse_matrix[i] = std::make_shared<TrilinosWrappers::SparseMatrix>();
          compute_sparse_matrix(i, *block_sparse_matrix[i]);
      }
      return block_sparse_matrix;
  }

private:
  const Discretization<dim, Number>    &discretization;

  double ghost_parameter_M;

  mutable std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> block_sparse_matrix;

  void
  compute_sparse_matrix(size_t domain_idx, TrilinosWrappers::SparseMatrix &mat) const
  {
    AssertThrow(ghost_parameter_M != -1.0, ExcNotImplemented());

    // 0) extract information from discretization class
    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const Quadrature<1> &quadrature_1D = discretization.get_quadrature_1D();
    const Quadrature<dim - 1> &face_quadrature =
      discretization.get_face_quadrature();
    const GDM::System<dim>          &system = discretization.get_system();
    const AffineConstraints<Number> &constraints =
      discretization.get_affine_constraints();
    const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> &mesh_classifiers =
      discretization.get_mesh_classifiers();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const std::vector<VectorType>              &level_sets = discretization.get_level_sets();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    // 1) create sparsity pattern
    if (mat.m() == 0 || mat.n() == 0)
      {
        TrilinosWrappers::SparsityPattern sparsity_pattern;
        sparsity_pattern.reinit(system.locally_owned_dofs(), MPI_COMM_WORLD);
        system.create_flux_sparsity_pattern(constraints, sparsity_pattern);
        sparsity_pattern.compress();

        mat.reinit(sparsity_pattern);
      }
    else
      {
        mat = 0.0;
      }

    const auto face_has_ghost_penalty = [&](const auto        &cell,
                                            const unsigned int face_index) {
      if (cell->at_boundary(face_index))
        return false;

      const NonMatching::LocationToLevelSet cell_location =
        mesh_classifiers[domain_idx]->location_to_level_set(cell);

      const NonMatching::LocationToLevelSet neighbor_location =
        mesh_classifiers[domain_idx]->location_to_level_set(cell->neighbor(face_index));

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

    hp::FECollection<dim> fe_collection(fe);
    hp::QCollection<dim>  quadrature_collection(quadrature_1D);
    hp::QCollection<1>    quadrature_1D_collection(quadrature_1D);

    NonMatching::FEValues<dim> non_matching_fe_values(mapping,
                                                      fe_collection,
                                                      quadrature_collection,
                                                      quadrature_1D_collection,
                                                      region_update_flags,
                                                      *mesh_classifiers[domain_idx],
                                                      level_set_dof_handler,
                                                      level_sets[domain_idx]);

    FEInterfaceValues<dim> fe_interface_values(
      mapping,
      fe,
      hp::QCollection<dim - 1>(face_quadrature),
      update_gradients | update_JxW_values | update_normal_vectors);

    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : system.locally_active_cell_iterators())
      if (cell->is_locally_owned() &&
          (mesh_classifiers[domain_idx]->location_to_level_set(cell->dealii_iterator()) !=
           NonMatching::LocationToLevelSet::outside))
        {
          non_matching_fe_values.reinit(cell->dealii_iterator(),
                                        numbers::invalid_unsigned_int,
                                        numbers::invalid_unsigned_int,
                                        cell->active_fe_index());

          const double cell_side_length =
            cell->dealii_iterator()->minimum_vertex_distance();

          const auto &fe_values = non_matching_fe_values.get_inside_fe_values();

          const unsigned int dofs_per_cell = fe[0].dofs_per_cell;

          // compute element stiffness matrix
          FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);

          // (I) cell integral:
          if (fe_values)
            {
              for (const unsigned int q_index :
                   fe_values->quadrature_point_indices())
                {
                  for (const unsigned int i : fe_values->dof_indices())
                    for (const unsigned int j : fe_values->dof_indices())
                      // (v, u)
                      cell_matrix(i, j) += fe_values->shape_value(i, q_index) *
                                           fe_values->shape_value(j, q_index) *
                                           fe_values->JxW(q_index);
                }
            }

          // (II) face integral to apply GP:
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
                          // γ_M j(v, u) with j(v, u)= ∑ h^3 <∂v/∂n, ∂u/∂n>
                          local_stabilization(i, j) +=
                            .5 * ghost_parameter_M * cell_side_length *
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

                mat.add(local_interface_dof_indices,
                                  local_stabilization);
              }

          // get indices
          dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          // assemble
          constraints.distribute_local_to_global(cell_matrix,
                                                 dof_indices,
                                                 mat);
        }

    mat.compress(VectorOperation::values::add);

    for (auto &entry : mat)
      if ((entry.row() == entry.column()) && (entry.value() == 0.0))
        entry.value() = 1.0;
  }
};
