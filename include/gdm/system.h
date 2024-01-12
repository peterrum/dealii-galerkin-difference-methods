#pragma once

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>

#include <gdm/fe.h>

using namespace dealii;

namespace GDM
{
  template <int dim>
  class System;


  namespace internal
  {
    template <int dim>
    class CellAccessor
    {
    public:
      CellAccessor(const System<dim> &system, const unsigned int _index)
        : system(system)
        , _index(_index)
      {}

      unsigned int
      index() const
      {
        return _index;
      }

      unsigned int
      active_fe_index() const
      {
        return system.active_fe_indices[_index];
      }

      void
      operator++()
      {
        _index++;
      }

      void
      operator++(int)
      {
        _index++;
      }

      void
      operator--()
      {
        _index--;
      }

      void
      operator--(int)
      {
        _index--;
      }

      bool
      operator==(const CellAccessor &other) const
      {
        return _index == other._index;
      }

      bool
      operator!=(const CellAccessor &other) const
      {
        return _index != other._index;
      }

      typename Triangulation<dim>::active_cell_iterator
      dealii_iterator() const
      {
        return typename Triangulation<dim>::active_cell_iterator(&system.tria,
                                                                 0,
                                                                 _index);
      }

      void
      get_dof_indices(std::vector<types::global_dof_index> &dof_indices) const
      {
        const auto indices =
          index_to_indices<dim>(_index, system.n_subdivisions);

        std::array<unsigned int, dim> offset_reference;
        for (unsigned int d = 0; d < dim; ++d)
          offset_reference[d] =
            (indices[d] < system.fe_degree / 2) ?
              0 :
              (std::min(system.n_subdivisions[d],
                        indices[d] + system.fe_degree / 2 + 1) -
               system.fe_degree);

        std::array<unsigned int, dim> n_dofs;
        for (unsigned int d = 0; d < dim; ++d)
          n_dofs[d] = system.n_subdivisions[d] + 1;

        for (unsigned int k = 0, c = 0;
             k <= ((dim >= 3) ? system.fe_degree : 0);
             ++k)
          for (unsigned int j = 0; j <= ((dim >= 2) ? system.fe_degree : 0);
               ++j)
            for (unsigned int i = 0; i <= system.fe_degree; ++i, ++c)
              {
                auto offset = offset_reference;

                if (dim >= 1)
                  offset[0] += i;
                if (dim >= 2)
                  offset[1] += j;
                if (dim >= 3)
                  offset[2] += k;

                const auto index = indices_to_index<dim>(offset, n_dofs);

                AssertIndexRange(index, system.n_dofs());

                dof_indices[c] = index;
              }
      }

    private:
      const System<dim> &system;
      unsigned int       _index;
    };



    template <int dim>
    class CellIterator
    {
    public:
      using value_type      = CellAccessor<dim>;
      using difference_type = int;

      CellIterator(CellAccessor<dim> accessor)
        : accessor(accessor)
      {}

      const CellAccessor<dim> *
      operator->() const
      {
        return &accessor;
      }

      CellIterator &
      operator++()
      {
        accessor++;

        return *this;
      }

      CellIterator &
      operator++(int n)
      {
        accessor->operator++(n);

        return *this;
      }

      CellIterator &
      operator--()
      {
        accessor--;

        return *this;
      }

      CellIterator &
      operator--(int n)
      {
        accessor->operator--(n);

        return *this;
      }

      bool
      operator==(const CellIterator &other) const
      {
        return accessor == other.accessor;
      }

      bool
      operator!=(const CellIterator &other) const
      {
        return accessor != other.accessor;
      }

    private:
      CellAccessor<dim> accessor;
    };
  } // namespace internal


  template <int dim>
  class System
  {
  public:
    System(const unsigned int fe_degree, const unsigned int n_components)
      : fe_degree(fe_degree)
      , fe(generate_fe_collection<dim>(generate_polynomials_1D(fe_degree),
                                       n_components))
    {}


    void
    subdivided_hyper_cube(const unsigned int n_subdivisions_1D)
    {
      std::fill(this->n_subdivisions.begin(),
                this->n_subdivisions.end(),
                n_subdivisions_1D);

      GridGenerator::subdivided_hyper_cube(tria, n_subdivisions_1D);
    }


    void
    categorize()
    {
      active_fe_indices.resize(tria.n_active_cells());

      for (const auto &cell : tria.active_cell_iterators())
        {
          unsigned int cell_index = cell->active_cell_index(); // TODO: better?

          auto indices = index_to_indices<dim>(cell_index, n_subdivisions);

          for (unsigned int d = 0; d < dim; ++d)
            indices[d] =
              (indices[d] < (fe_degree / 2) ?
                 indices[d] :
                 (indices[d] < (n_subdivisions[d] - fe_degree / 2) ?
                    (fe_degree / 2) :
                    (2 + indices[d] + fe_degree / 2 - n_subdivisions[d])));

          active_fe_indices[cell->active_cell_index()] =
            indices_to_index<dim>(indices, fe_degree);
        }
    }


    template <typename Number>
    void
    fill_constraints(AffineConstraints<Number> &constraints) const
    {
      for (unsigned int surface = 0; surface < 2 * dim; ++surface)
        {
          const unsigned int d = surface / 2; // direction
          const unsigned int s = surface % 2; // left or right surface

          unsigned int n0 = 1;
          for (unsigned int i = d + 1; i < dim; ++i)
            n0 *= n_subdivisions[i] + 1;

          unsigned int n1 = 1;
          for (unsigned int i = 0; i < d; ++i)
            n1 *= n_subdivisions[i] + 1;

          unsigned int n2 = n1 * (n_subdivisions[d] + 1);

          for (unsigned int i = 0; i < n0; ++i)
            for (unsigned int j = 0; j < n1; ++j)
              constraints.constrain_dof_to_zero(
                i * n2 + (s == 0 ? 0 : n_subdivisions[d]) * n1 + j);
        }
    }


    const hp::FECollection<dim> &
    get_fe() const
    {
      return fe;
    }


    const Triangulation<dim> &
    get_triangulation() const
    {
      return tria;
    }

    types::global_dof_index
    n_dofs() const
    {
      types::global_dof_index n = 1;

      for (unsigned int d = 0; d < dim; ++d)
        n *= n_subdivisions[d] + 1;

      return n;
    }


    template <typename SparsityPatternType>
    void
    create_sparsity_pattern(SparsityPatternType &dsp) const
    {
      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : active_cell_iterators())
        {
          dof_indices.resize(fe[cell->active_fe_index()].n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);

          for (const auto i : dof_indices)
            dsp.add_entries(i, dof_indices.begin(), dof_indices.end());
        }
    }


    IteratorRange<GDM::internal::CellIterator<dim>>
    active_cell_iterators() const
    {
      return {GDM::internal::CellIterator<dim>(
                GDM::internal::CellAccessor<dim>(*this, 0)),
              GDM::internal::CellIterator<dim>(
                GDM::internal::CellAccessor<dim>(*this, tria.n_cells()))};
    }

    IndexSet
    locally_owned_dofs()
    {
      IndexSet is(n_dofs());
      is.add_range(0, n_dofs());

      return is;
    }

  private:
    // finite element
    const unsigned int          fe_degree;
    const hp::FECollection<dim> fe;

    // geometry
    std::array<unsigned int, dim> n_subdivisions;
    Triangulation<dim>            tria;

    // category
    std::vector<unsigned int> active_fe_indices;

    friend GDM::internal::CellAccessor<dim>;
  };

} // namespace GDM
