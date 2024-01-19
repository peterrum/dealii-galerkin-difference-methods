#pragma once

#include <gdm/system.h>

using namespace dealii;

namespace GDM
{
  namespace VectorTools
  {
    template <typename VectorType, int dim, int spacedim>
    void
    interpolate(
      const hp::MappingCollection<dim, spacedim>                &mapping,
      const GDM::System<dim>                                    &system,
      const Function<spacedim, typename VectorType::value_type> &function,
      VectorType                                                &vec)
    {
      dealii::VectorTools::interpolate(mapping,
                                       system.get_dof_handler(),
                                       function,
                                       vec);
    }

    template <int dim, typename Number, class OutVector>
    void
    integrate_difference(const hp::MappingCollection<dim>    &mapping,
                         const System<dim>                   &system,
                         const ReadVector<Number>            &fe_function,
                         const Function<dim, Number>         &exact_solution,
                         OutVector                           &difference,
                         const hp::QCollection<dim>          &quadrature,
                         const dealii::VectorTools::NormType &norm)
    {
      AssertDimension(dealii::VectorTools::NormType::L2_norm, norm);

      difference.reinit(system.get_triangulation().n_active_cells());

      hp::FEValues<dim> fe_values_collection(mapping,
                                             system.get_fe(),
                                             quadrature,
                                             update_quadrature_points |
                                               update_values |
                                               update_JxW_values);

      std::vector<types::global_dof_index> dof_indices;
      std::vector<Number>                  values;
      std::vector<Number>                  values_exact;
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

          values.resize(fe_values.n_quadrature_points);
          fe_values.get_function_values(fe_function, dof_indices, values);

          values_exact.resize(fe_values.n_quadrature_points);
          exact_solution.value_list(fe_values.get_quadrature_points(),
                                    values_exact);

          Number diff = 0.0;

          for (const auto q : fe_values.dof_indices())
            diff +=
              Utilities::pow(values[q] - values_exact[q], 2) * fe_values.JxW(q);

          difference[cell->dealii_iterator()->active_cell_index()] =
            std::sqrt(diff);
        }
    }

  } // namespace VectorTools
} // namespace GDM
