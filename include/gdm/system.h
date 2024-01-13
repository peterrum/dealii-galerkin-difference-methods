#pragma once

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/shared_tria.h>

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
        return typename Triangulation<dim>::active_cell_iterator(
          &system.get_triangulation(), 0, system.active_cell_index_map[_index]);
      }

      void
      get_dof_indices(std::vector<types::global_dof_index> &dof_indices) const
      {
        const auto indices =
          index_to_indices<dim>(system.active_cell_index_map[_index],
                                system.n_subdivisions);

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
      : comm(MPI_COMM_NULL)
      , fe_degree(fe_degree)
      , fe(generate_fe_collection<dim>(generate_polynomials_1D(fe_degree),
                                       n_components))
    {}

    System(const MPI_Comm     comm,
           const unsigned int fe_degree,
           const unsigned int n_components)
      : comm(comm)
      , fe_degree(fe_degree)
      , fe(generate_fe_collection<dim>(generate_polynomials_1D(fe_degree),
                                       n_components))
    {}


    void
    subdivided_hyper_cube(const unsigned int n_subdivisions_1D)
    {
      std::fill(this->n_subdivisions.begin(),
                this->n_subdivisions.end(),
                n_subdivisions_1D);

      if (comm == MPI_COMM_NULL)
        {
          tria = std::make_shared<Triangulation<dim>>();

          unsigned int dofs = 1;
          for (unsigned int d = 0; d < dim; ++d)
            dofs *= n_subdivisions[d] + 1;

          IndexSet is_local(dofs);
          is_local.add_range(0, dofs);
          this->is_local = is_local;
        }
      else
        {
          const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
          const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

          unsigned int face_dofs = 1;
          for (unsigned int d = 0; d < dim - 1; ++d)
            face_dofs *= n_subdivisions[d] + 1;

          IndexSet is_local(face_dofs * (n_subdivisions[dim - 1] + 1));

          const unsigned int stride =
            (n_subdivisions[dim - 1] + n_procs - 1) / n_procs;
          unsigned int range_start =
            (my_rank == 0) ? 0 : ((stride * my_rank) + 1);
          unsigned int range_end = stride * (my_rank + 1) + 1;

          is_local.add_range(
            face_dofs * std::min(range_start, n_subdivisions[dim - 1] + 1),
            face_dofs * std::min(range_end, n_subdivisions[dim - 1] + 1));

          this->is_local = is_local;

          auto temp = std::make_shared<parallel::shared::Triangulation<dim>>(
            comm,
            Triangulation<dim>::none,
            true,
            parallel::shared::Triangulation<dim>::partition_custom_signal);

          temp->signals.create.connect([&, stride]() {
            for (const auto &cell : tria->active_cell_iterators())
              {
                unsigned int cell_index = cell->active_cell_index();

                auto indices =
                  index_to_indices<dim>(cell_index, n_subdivisions);

                cell->set_subdomain_id(indices[dim - 1] / stride);
              }
          });

          tria = temp;
        }

      GridGenerator::subdivided_hyper_cube(*tria, n_subdivisions_1D);

      for (const auto &cell : tria->active_cell_iterators())
        if (cell->is_locally_owned())
          active_cell_index_map.push_back(cell->active_cell_index());

      if (comm == MPI_COMM_NULL)
        {
          this->is_ghost = this->is_local;
        }
      else
        {
          IndexSet is_ghost = this->is_local;

          std::vector<types::global_dof_index> dof_indices;
          for (const auto &cell : active_cell_iterators())
            {
              dof_indices.resize(fe[0 /*TODO*/].n_dofs_per_cell());
              cell->get_dof_indices(dof_indices);

              for (const auto i : dof_indices)
                if (is_local.is_element(i) == false)
                  is_ghost.add_index(i);
            }

          this->is_ghost = is_ghost;
        }
    }


    void
    categorize()
    {
      active_fe_indices.resize(active_cell_index_map.size());

      for (unsigned int i = 0; i < active_cell_index_map.size(); ++i)
        {
          unsigned int cell_index = active_cell_index_map[i];

          auto indices = index_to_indices<dim>(cell_index, n_subdivisions);

          for (unsigned int d = 0; d < dim; ++d)
            indices[d] =
              (indices[d] < (fe_degree / 2) ?
                 indices[d] :
                 (indices[d] < (n_subdivisions[d] - fe_degree / 2) ?
                    (fe_degree / 2) :
                    (2 + indices[d] + fe_degree / 2 - n_subdivisions[d])));

          active_fe_indices[i] = indices_to_index<dim>(indices, fe_degree);
        }
    }


    template <typename Number>
    void
    make_zero_boundary_constraints(const unsigned int         surface,
                                   AffineConstraints<Number> &constraints) const
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


    template <typename Number>
    void
    make_zero_boundary_constraints(AffineConstraints<Number> &constraints) const
    {
      for (unsigned int surface = 0; surface < 2 * dim; ++surface)
        make_zero_boundary_constraints(surface, constraints);
    }


    const hp::FECollection<dim> &
    get_fe() const
    {
      return fe;
    }


    const Triangulation<dim> &
    get_triangulation() const
    {
      return *tria;
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
              GDM::internal::CellIterator<dim>(GDM::internal::CellAccessor<dim>(
                *this, active_cell_index_map.size()))};
    }

    IndexSet
    locally_owned_dofs()
    {
      return is_local;
    }

    IndexSet
    locally_active_dofs()
    {
      return is_ghost;
    }

  private:
    const MPI_Comm comm;

    // finite element
    const unsigned int          fe_degree;
    const hp::FECollection<dim> fe;

    // geometry
    std::array<unsigned int, dim>       n_subdivisions;
    std::shared_ptr<Triangulation<dim>> tria;

    //
    IndexSet is_local;
    IndexSet is_ghost;

    std::vector<unsigned int> active_cell_index_map;

    // category
    std::vector<unsigned int> active_fe_indices;

    friend GDM::internal::CellAccessor<dim>;
  };

} // namespace GDM
