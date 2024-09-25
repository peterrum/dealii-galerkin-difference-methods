#pragma once

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include "wave-discretization.h"

using namespace dealii;

template <unsigned int dim, typename Number>
class StiffnessMatrixOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  StiffnessMatrixOperator(const Discretization<dim, Number> &discretization)
    : discretization(discretization)
  {}

  const TrilinosWrappers::SparseMatrix &
  get_sparse_matrix() const
  {
    compute_sparse_matrix();

    return sparse_matrix;
  }

private:
  const Discretization<dim, Number> &discretization;

  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;

  void
  compute_sparse_matrix() const
  {
    const double ghost_parameter_A = 0.50 * std::sqrt(3.0); // TODO

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

          const auto &fe_values = non_matching_fe_values.get_inside_fe_values();

          const unsigned int dofs_per_cell = fe[0].dofs_per_cell;

          // compute element stiffness matrix
          FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);

          // (I) cell integral
          if (fe_values)
            {
              for (const unsigned int q_index :
                   fe_values->quadrature_point_indices())
                {
                  for (const unsigned int i : fe_values->dof_indices())
                    for (const unsigned int j : fe_values->dof_indices())
                      cell_matrix(i, j) += fe_values->shape_grad(i, q_index) *
                                           fe_values->shape_grad(j, q_index) *
                                           fe_values->JxW(q_index);
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
