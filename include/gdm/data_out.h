#pragma once

#include <deal.II/numerics/data_out.h>

#include <gdm/system.h>

namespace GDM
{

  template <int dim>
  class DataOut
  {
  public:
    DataOut(const System<dim>                &system,
            const hp::MappingCollection<dim> &mapping,
            const unsigned int                fe_degree_output)
      : system(system)
      , dof_handler_output(system.get_triangulation())
      , mapping(mapping)
      , fe_degree_output(fe_degree_output)
    {
      FE_DGQ<dim> fe_output(fe_degree_output);
      dof_handler_output.distribute_dofs(fe_output);

      data_out.attach_dof_handler(dof_handler_output);
    }

    template <typename VectorType>
    void
    add_data_vector(const VectorType &solution, const std::string label)
    {
      VectorType solution_output(dof_handler_output.n_dofs());

      hp::QCollection<dim> quadrature;
      quadrature.push_back(
        Quadrature<dim>(dof_handler_output.get_fe().get_unit_support_points()));

      hp::FEValues<dim> fe_values_collection(mapping,
                                             system.get_fe(),
                                             quadrature,
                                             update_gradients | update_values |
                                               update_JxW_values);

      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : system.active_cell_iterators())
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

          // read vector
          Vector<double> cell_vector_input(dofs_per_cell);

          for (const unsigned int i : fe_values.dof_indices())
            cell_vector_input[i] = solution[dof_indices[i]];

          // perform interpolation
          Vector<double> cell_vector_output(
            dof_handler_output.get_fe().n_dofs_per_cell());

          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            for (const unsigned int i : fe_values.dof_indices())
              cell_vector_output(q_index) +=
                fe_values.shape_value(i, q_index) * cell_vector_input[i];

          // write
          cell->dealii_iterator()
            ->as_dof_handler_iterator(dof_handler_output)
            ->set_dof_values(cell_vector_output, solution_output);
        }

      data_out.add_data_vector(solution_output, label);
    }


    void
    build_patches()
    {
      data_out.build_patches(mapping, fe_degree_output);
    }

    void
    write_vtu(std::ostream &out) const
    {
      data_out.write_vtu(out);
    }

  private:
    const System<dim> &system;
    DoFHandler<dim>    dof_handler_output;

    const hp::MappingCollection<dim> &mapping;
    const unsigned int                fe_degree_output;

    dealii::DataOut<dim> data_out;
  };

} // namespace GDM
