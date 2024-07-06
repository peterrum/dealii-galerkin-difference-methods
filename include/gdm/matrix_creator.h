#pragma once

#include <gdm/system.h>

namespace GDM
{
  namespace MatrixCreator
  {
    template <int dim, typename SparseMatrixType>
    void
    create_mass_matrix(
      const hp::MappingCollection<dim> &mapping,
      const System<dim>                &system,
      const hp::QCollection<dim>       &quadrature,
      SparseMatrixType                 &sparse_matrix,
      const AffineConstraints<typename SparseMatrixType::value_type>
        &constraints)
    {
      using Number = typename SparseMatrixType::value_type;

      hp::FEValues<dim> fe_values_collection(mapping,
                                             system.get_fe(),
                                             quadrature,
                                             update_values | update_JxW_values);

      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : system.locally_active_cell_iterators())
        {
          fe_values_collection.reinit(cell->dealii_iterator(),
                                      numbers::invalid_unsigned_int,
                                      numbers::invalid_unsigned_int,
                                      cell->active_fe_index());

          const auto &fe_values = fe_values_collection.get_present_fe_values();

          const unsigned int dofs_per_cell =
            fe_values.get_fe().n_dofs_per_cell();

          // get indices
          dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          // compute element stiffness matrix
          FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                       fe_values.shape_value(j, q_index) *
                                       fe_values.JxW(q_index);
            }

          // assemble
          constraints.distribute_local_to_global(cell_matrix,
                                                 dof_indices,
                                                 sparse_matrix);
        }

      sparse_matrix.compress(VectorOperation::values::add);
    }

    template <int dim, typename VectorType>
    void
    create_lumped_mass_matrix(
      const hp::MappingCollection<dim>                         &mapping,
      const System<dim>                                        &system,
      const hp::QCollection<dim>                               &quadrature,
      VectorType                                               &vector,
      const AffineConstraints<typename VectorType::value_type> &constraints)
    {
      using Number = typename VectorType::value_type;

      hp::FEValues<dim> fe_values_collection(mapping,
                                             system.get_fe(),
                                             quadrature,
                                             update_values | update_JxW_values);

      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : system.locally_active_cell_iterators())
        {
          fe_values_collection.reinit(cell->dealii_iterator(),
                                      numbers::invalid_unsigned_int,
                                      numbers::invalid_unsigned_int,
                                      cell->active_fe_index());

          const auto &fe_values = fe_values_collection.get_present_fe_values();

          const unsigned int dofs_per_cell =
            fe_values.get_fe().n_dofs_per_cell();

          // get indices
          dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          // compute element stiffness matrix
          Vector<Number> cell_vector(dofs_per_cell);
          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                  cell_vector(i) += fe_values.shape_value(i, q_index) *
                                    fe_values.shape_value(j, q_index) *
                                    fe_values.JxW(q_index);
            }

          // assemble
          constraints.distribute_local_to_global(cell_vector,
                                                 dof_indices,
                                                 vector);
        }

      for (auto &entry : vector)
        entry = Number(1.0) / entry;
    }
  } // namespace MatrixCreator
} // namespace GDM
