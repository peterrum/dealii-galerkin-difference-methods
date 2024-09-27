#pragma once

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/vector_tools.h>

#include <gdm/fe.h>

using namespace dealii;

namespace GDM
{
  template <int dim>
  class System;


  namespace internal
  {
    template <int dim>
    class CellAccessor;



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



    template <int dim>
    class CellAccessor
    {
    public:
      CellAccessor(const System<dim> &system, const unsigned int _index)
        : system(system)
        , _index(_index)
      {
        AssertIndexRange(_index, system.active_cell_index_map.size() + 1);
      }

      unsigned int
      index() const
      {
        return _index;
      }

      unsigned int
      active_fe_index() const
      {
        AssertIndexRange(_index, system.active_fe_indices.size());

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

      CellIterator<dim>
      neighbor(const unsigned int f) const
      {
        auto index_neighbor = _index;

        if (f == 0)
          index_neighbor -= 1;
        else if (f == 1)
          index_neighbor += 1;
        else if (f == 2)
          index_neighbor -= system.n_subdivisions[0];
        else if (f == 3)
          index_neighbor += system.n_subdivisions[1];
        else
          Assert(false, ExcNotImplemented());

        return CellIterator<dim>(CellAccessor(this->system, index_neighbor));
      }

      bool
      is_locally_owned() const
      {
        return dealii_iterator()->is_locally_owned();
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
    void
    compute_renumbering_lex(dealii::DoFHandler<dim> &dof_handler)
    {
      std::vector<dealii::types::global_dof_index> dof_indices(
        dof_handler.get_fe().n_dofs_per_cell());

      dealii::IndexSet active_dofs;
      dealii::DoFTools::extract_locally_active_dofs(dof_handler, active_dofs);
      const auto partitioner =
        std::make_shared<dealii::Utilities::MPI::Partitioner>(
          dof_handler.locally_owned_dofs(), active_dofs, MPI_COMM_WORLD);

      std::vector<
        std::pair<dealii::types::global_dof_index, dealii::Point<dim>>>
        points_all;

      dealii::FEValues<dim> fe_values(
        dof_handler.get_fe(),
        dealii::Quadrature<dim>(dof_handler.get_fe().get_unit_support_points()),
        dealii::update_quadrature_points);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          fe_values.reinit(cell);

          cell->get_dof_indices(dof_indices);

          for (unsigned int i = 0; i < dof_indices.size(); ++i)
            {
              if (dof_handler.locally_owned_dofs().is_element(dof_indices[i]))
                points_all.emplace_back(dof_indices[i],
                                        fe_values.quadrature_point(i));
            }
        }

      std::sort(points_all.begin(),
                points_all.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      points_all.erase(std::unique(points_all.begin(),
                                   points_all.end(),
                                   [](const auto &a, const auto &b) {
                                     return a.first == b.first;
                                   }),
                       points_all.end());

      std::sort(points_all.begin(),
                points_all.end(),
                [](const auto &a, const auto &b) {
                  std::vector<double> a_(dim);
                  std::vector<double> b_(dim);

                  a.second.unroll(a_.begin(), a_.end());
                  std::reverse(a_.begin(), a_.end());

                  b.second.unroll(b_.begin(), b_.end());
                  std::reverse(b_.begin(), b_.end());

                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      if (std::abs(a_[d] - b_[d]) > 1e-8 /*epsilon*/)
                        return a_[d] < b_[d];
                    }

                  return true;
                });

      std::vector<dealii::types::global_dof_index> result(
        dof_handler.n_locally_owned_dofs());

      for (unsigned int i = 0; i < result.size(); ++i)
        {
          result[partitioner->global_to_local(points_all[i].first)] =
            partitioner->local_to_global(i);
        }

      dof_handler.renumber_dofs(result);
    }
  } // namespace internal


  template <int dim>
  class System
  {
  public:
    System(const unsigned int fe_degree,
           const unsigned int n_components,
           const bool         add_ghost_layer = false)
      : comm(MPI_COMM_NULL)
      , fe_degree(fe_degree)
      , fe(generate_fe_collection<dim>(generate_polynomials_1D(fe_degree),
                                       n_components))
      , add_ghost_layer(add_ghost_layer)
    {}

    System(const MPI_Comm     comm,
           const unsigned int fe_degree,
           const unsigned int n_components,
           const bool         add_ghost_layer = false)
      : comm(comm)
      , fe_degree(fe_degree)
      , fe(generate_fe_collection<dim>(generate_polynomials_1D(fe_degree),
                                       n_components))
      , add_ghost_layer(add_ghost_layer)
    {}


    void
    subdivided_hyper_cube(const unsigned int n_subdivisions_1D,
                          const double       left  = 0.0,
                          const double       right = 1.0)
    {
      std::fill(this->n_subdivisions.begin(),
                this->n_subdivisions.end(),
                n_subdivisions_1D);

      create_triangulation_pre();

      GridGenerator::subdivided_hyper_cube(
        *tria, n_subdivisions_1D, left, right, true);

      create_triangulation_post();
    }


    void
    subdivided_hyper_rectangle(const std::vector<unsigned int> &repetitions,
                               const Point<dim>                &p1,
                               const Point<dim>                &p2)
    {
      AssertDimension(repetitions.size(), dim);

      for (unsigned int d = 0; d < dim; ++d)
        this->n_subdivisions[d] = repetitions[d];

      create_triangulation_pre();

      GridGenerator::subdivided_hyper_rectangle(
        *tria, repetitions, p1, p2, true);

      create_triangulation_post();
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
            indices[d] = (indices[d] < (fe_degree / 2) ?
                            indices[d] :
                            (indices[d] < (n_subdivisions[d] - fe_degree / 2) ?
                               (fe_degree / 2) :
                               (fe_degree + indices[d] - n_subdivisions[d])));

          active_fe_indices[i] = indices_to_index<dim>(indices, fe_degree);
        }
    }


    template <typename Number>
    void
    make_periodicity_constraints(const unsigned int         d,
                                 AffineConstraints<Number> &constraints) const
    {
      unsigned int n0 = 1;
      for (unsigned int i = d + 1; i < dim; ++i)
        n0 *= n_subdivisions[i] + 1;

      unsigned int n1 = 1;
      for (unsigned int i = 0; i < d; ++i)
        n1 *= n_subdivisions[i] + 1;

      const unsigned int n2 = n1 * (n_subdivisions[d] + 1);

      for (unsigned int i = 0; i < n0; ++i)
        for (unsigned int j = 0; j < n1; ++j)
          {
            const unsigned int i0 = i * n2 + j;
            const unsigned int i1 = i0 + n_subdivisions[d] * n1;

            if (is_locally_active.is_element(i1) == false)
              continue;

            if (constraints.is_constrained(i1))
              continue;

            constraints.add_line(i1);
            constraints.add_entry(i1, i0, 1.0);
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

      const unsigned int n2 = n1 * (n_subdivisions[d] + 1);

      for (unsigned int i = 0; i < n0; ++i)
        for (unsigned int j = 0; j < n1; ++j)
          {
            const unsigned i0 =
              i * n2 + (s == 0 ? 0 : n_subdivisions[d]) * n1 + j;

            if (is_locally_active.is_element(i0) == false)
              continue;

            constraints.constrain_dof_to_zero(i0);
          }
    }


    template <typename Number>
    void
    make_zero_boundary_constraints(AffineConstraints<Number> &constraints) const
    {
      for (unsigned int surface = 0; surface < 2 * dim; ++surface)
        make_zero_boundary_constraints(surface, constraints);
    }


    template <typename Number>
    void
    interpolate_boundary_values(const hp::MappingCollection<dim> &mapping,
                                const unsigned int                bid,
                                const Function<dim>              &fu,
                                AffineConstraints<Number> &constraints) const
    {
      if (comm == MPI_COMM_NULL)
        {
          VectorTools::interpolate_boundary_values(
            mapping, dof_handler, bid, fu, constraints);
        }
      else
        {
          AffineConstraints<Number> constraints_temp(is_locally_active);
          VectorTools::interpolate_boundary_values(
            mapping, dof_handler, bid, fu, constraints_temp);
          constraints_temp.make_consistent_in_parallel(is_locally_owned,
                                                       is_locally_active,
                                                       comm);
          constraints_temp.close();

          for (const auto i : is_locally_active)
            {
              if ((constraints_temp.is_constrained(i) == false) ||
                  (constraints.is_constrained(i) == true))
                continue;

              const Number inhomogeneity =
                constraints_temp.is_inhomogeneously_constrained(i) ?
                  constraints_temp.get_inhomogeneity(i) :
                  0.0;

              constraints.add_constraint(i, {}, inhomogeneity);
            }
        }
    }


    template <typename Number>
    void
    interpolate_boundary_values(const hp::MappingCollection<dim> &mapping,
                                const Function<dim>              &fu,
                                AffineConstraints<Number> &constraints) const
    {
      for (unsigned int surface = 0; surface < 2 * dim; ++surface)
        interpolate_boundary_values(mapping, fu, constraints);
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


    template <typename Number, typename SparsityPatternType>
    void
    create_sparsity_pattern(const AffineConstraints<Number> &constraints,
                            SparsityPatternType             &dsp) const
    {
      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : locally_active_cell_iterators())
        {
          dof_indices.resize(fe[cell->active_fe_index()].n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);

          constraints.add_entries_local_to_global(dof_indices, dsp);
        }
    }


    template <typename Number, typename SparsityPatternType>
    void
    create_flux_sparsity_pattern(const AffineConstraints<Number> &constraints,
                                 SparsityPatternType             &dsp) const
    {
      Assert(add_ghost_layer, ExcNotImplemented());

      std::vector<types::global_dof_index> dof_indices;
      std::vector<types::global_dof_index> dof_indices_neighbor;
      for (const auto &cell : locally_active_cell_iterators())
        if (cell->is_locally_owned())
          {
            dof_indices.resize(fe[cell->active_fe_index()].n_dofs_per_cell());
            cell->get_dof_indices(dof_indices);

            constraints.add_entries_local_to_global(dof_indices, dsp);

            for (unsigned int i = 0; i < 2 * dim; ++i)
              if (cell->dealii_iterator()->at_boundary(i) == false)
                {
                  dof_indices_neighbor.resize(
                    fe[cell->neighbor(i)->active_fe_index()].n_dofs_per_cell());
                  cell->neighbor(i)->get_dof_indices(dof_indices_neighbor);
                  constraints.add_entries_local_to_global(dof_indices,
                                                          dof_indices_neighbor,
                                                          dsp);
                }
          }
    }


    IteratorRange<GDM::internal::CellIterator<dim>>
    locally_active_cell_iterators() const
    {
      return {GDM::internal::CellIterator<dim>(
                GDM::internal::CellAccessor<dim>(*this, 0)),
              GDM::internal::CellIterator<dim>(GDM::internal::CellAccessor<dim>(
                *this, active_cell_index_map.size()))};
    }


    IndexSet
    locally_owned_dofs() const
    {
      return this->is_locally_owned;
    }


    IndexSet
    locally_active_dofs() const
    {
      return this->is_locally_active;
    }


    template <typename Number>
    IndexSet
    locally_relevant_dofs(const AffineConstraints<Number> &constraints) const
    {
      IndexSet is_locally_relevant = this->is_locally_active;

      std::vector<types::global_dof_index> is_locally_relevant_temp;

      for (const auto i : is_locally_relevant)
        {
          if (is_locally_relevant.is_element(i) == false)
            is_locally_relevant_temp.emplace_back(i);

          const auto constraints_i = constraints.get_constraint_entries(i);

          if (constraints_i)
            for (const auto &p : *constraints_i)
              if (is_locally_relevant.is_element(p.first) == false)
                is_locally_relevant_temp.emplace_back(p.first);
        }

      std::sort(is_locally_relevant_temp.begin(),
                is_locally_relevant_temp.end());
      is_locally_relevant_temp.erase(
        std::unique(is_locally_relevant_temp.begin(),
                    is_locally_relevant_temp.end()),
        is_locally_relevant_temp.end());
      is_locally_relevant.add_indices(is_locally_relevant_temp.begin(),
                                      is_locally_relevant_temp.end());

      return is_locally_relevant;
    }

    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return dof_handler;
    }

    unsigned int
    get_fe_degree()
    {
      return fe_degree;
    }

  private:
    void
    create_triangulation_pre()
    {
      if (comm == MPI_COMM_NULL)
        {
          tria = std::make_shared<Triangulation<dim>>();

          unsigned int dofs = 1;
          for (unsigned int d = 0; d < dim; ++d)
            dofs *= n_subdivisions[d] + 1;

          IndexSet is_locally_owned(dofs);
          is_locally_owned.add_range(0, dofs);
          this->is_locally_owned = is_locally_owned;
        }
      else
        {
          const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
          const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

          unsigned int face_dofs = 1;
          for (unsigned int d = 0; d < dim - 1; ++d)
            face_dofs *= n_subdivisions[d] + 1;

          IndexSet is_locally_owned(face_dofs * (n_subdivisions[dim - 1] + 1));

          const unsigned int stride =
            (n_subdivisions[dim - 1] + n_procs - 1) / n_procs;
          unsigned int range_start =
            (my_rank == 0) ? 0 : ((stride * my_rank) + 1);
          unsigned int range_end = stride * (my_rank + 1) + 1;

          is_locally_owned.add_range(
            face_dofs * std::min(range_start, n_subdivisions[dim - 1] + 1),
            face_dofs * std::min(range_end, n_subdivisions[dim - 1] + 1));

          this->is_locally_owned = is_locally_owned;

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
    }


    void
    create_triangulation_post()
    {
      for (const auto &cell : tria->active_cell_iterators())
        if (cell->is_locally_owned())
          active_cell_index_map.push_back(cell->active_cell_index());
        else if ((add_ghost_layer) && (cell->is_artificial() == false))
          active_cell_index_map.push_back(cell->active_cell_index());

      if (comm == MPI_COMM_NULL)
        {
          this->is_locally_active = this->is_locally_owned;
        }
      else
        {
          IndexSet is_locally_active = this->is_locally_owned;

          std::vector<types::global_dof_index> dof_indices;
          for (const auto &cell : locally_active_cell_iterators())
            {
              dof_indices.resize(fe[0 /*TODO*/].n_dofs_per_cell());
              cell->get_dof_indices(dof_indices);

              for (const auto i : dof_indices)
                if (is_locally_owned.is_element(i) == false)
                  is_locally_active.add_index(i);
            }

          this->is_locally_active = is_locally_active;
        }

      dof_handler.reinit(*tria);
      dof_handler.distribute_dofs(FE_Q<dim>(1));
      internal::compute_renumbering_lex(dof_handler);
    }


    const MPI_Comm comm;

    // finite element
    const unsigned int          fe_degree;
    const hp::FECollection<dim> fe;

    const bool add_ghost_layer;

    // geometry
    std::array<unsigned int, dim>       n_subdivisions;
    std::shared_ptr<Triangulation<dim>> tria;

    DoFHandler<dim> dof_handler;

    // index sets
    IndexSet is_locally_owned;
    IndexSet is_locally_active;

    // map from local active cell index to active cell index
    std::vector<unsigned int> active_cell_index_map;

    // category
    std::vector<unsigned int> active_fe_indices;

    friend GDM::internal::CellAccessor<dim>;
  };

} // namespace GDM
