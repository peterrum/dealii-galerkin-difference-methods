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
  unsigned int
  get_category(const TriaActiveIterator<CellAccessor<dim>> &cell)
  {
    AssertDimension(dim, 1); // TODO: for higher dimension

    if (cell->at_boundary(0))
      return 0;
    else if (cell->at_boundary(1))
      return 2;
    else
      return 1;
  }


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
        const int offset =
          (active_fe_index() == 0) ? 0 : ((active_fe_index() == 1) ? -1 : -2);

        for (unsigned int i = 0;
             i < system.fe[active_fe_index()].n_dofs_per_cell();
             ++i)
          {
            dof_indices[i] = _index + i + offset;
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
      : fe(generate_fe_collection<dim>(generate_polynomials_1D(fe_degree),
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
        active_fe_indices[cell->active_cell_index()] = get_category(cell);
    }


    template <typename Number>
    void
    fill_constraints(AffineConstraints<Number> &constraints) const
    {
      AssertDimension(dim, 1); // TODO: higher dimensions

      constraints.constrain_dof_to_zero(0);
      constraints.constrain_dof_to_zero(n_subdivisions[0]);
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

    // finite element
    hp::FECollection<dim> fe;

    // geometry
    std::array<unsigned int, dim> n_subdivisions;
    Triangulation<dim>            tria;

    // category
    std::vector<unsigned int> active_fe_indices;

  private:
    friend GDM::internal::CellAccessor<dim>;
  };

} // namespace GDM
