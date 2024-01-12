#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

#include <gdm/fe.h>

#include <fstream>

using namespace dealii;

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
void
get_dof_indices(std::vector<types::global_dof_index>        &dof_indices,
                const TriaActiveIterator<CellAccessor<dim>> &cell,
                const hp::FECollection<dim>                 &fe,
                const unsigned int                           active_fe_index)
{
  AssertDimension(dim, 1); // TODO: for higher dimension/degree

  const unsigned int cell_index = cell->active_cell_index();

  const int offset =
    (active_fe_index == 0) ? 0 : ((active_fe_index == 1) ? -1 : -2);

  for (unsigned int i = 0; i < fe[active_fe_index].n_dofs_per_cell(); ++i)
    {
      dof_indices[i] = cell_index + i + offset;
    }
}



template <int dim>
class System
{
public:
  class CellAccessor
  {
  public:
    CellAccessor(const System<dim> &system, const unsigned int index)
      : system(system)
      , index(index)
    {}

    unsigned int
    active_fe_index() const;

    void
    operator++()
    {
      index++;
    }

    void
    operator++(int n)
    {
      index += n;
    }

    void
    operator--()
    {
      index--;
    }

    void
    operator--(int n)
    {
      index -= n;
    }

    bool
    operator==(const CellAccessor &other) const
    {
      return index == other.index;
    }

    bool
    operator!=(const CellAccessor &other) const
    {
      return index != other.index;
    }

    typename Triangulation<dim>::active_cell_iterator
    dealii_iterator() const;

    void
    get_dof_indices(std::vector<types::global_dof_index> &dof_indices) const;

  private:
    const System<dim> &system;
    unsigned int       index;
  };


  class CellIterator
  {
  public:
    using value_type      = CellAccessor;
    using difference_type = int;

    CellIterator(CellAccessor accessor)
      : accessor(accessor)
    {}

    const CellAccessor *
    operator->() const
    {
      return &accessor;
    }

    CellIterator
    operator++()
    {
      accessor++;

      return *this;
    }

    CellIterator
    operator++(int n)
    {
      accessor->operator++(n);

      return *this;
    }

    CellIterator
    operator--()
    {
      accessor--;

      return *this;
    }

    CellIterator
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
    CellAccessor accessor;
  };


  System(const unsigned int fe_degree)
    : fe(generate_fe_collection<dim>(generate_polynomials_1D(fe_degree)))
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
    for (const auto &cell : tria.active_cell_iterators())
      {
        const auto active_fe_index =
          active_fe_indices[cell->active_cell_index()];
        dof_indices.resize(fe[active_fe_index].n_dofs_per_cell());
        get_dof_indices(dof_indices, cell, fe, active_fe_index);

        for (const auto i : dof_indices)
          dsp.add_entries(i, dof_indices.begin(), dof_indices.end());
      }
  }


  IteratorRange<CellIterator>
  active_cell_iterators() const
  {
    return {CellIterator(CellAccessor(*this, 0)),
            CellIterator(CellAccessor(*this, tria.n_cells()))};
  }

  // finite element
  hp::FECollection<dim> fe;

  // geometry
  std::array<unsigned int, dim> n_subdivisions;
  Triangulation<dim>            tria;

  // category
  std::vector<unsigned int> active_fe_indices;

private:
};



template <int dim>
void
test()
{
  const unsigned int n_subdivisions   = 20;
  const unsigned int fe_degree        = 3;
  const unsigned int fe_degree_output = 2;

  // Create GDM system
  System<dim> system(fe_degree);

  // Create mesh
  system.subdivided_hyper_cube(n_subdivisions);

  // Create finite elements
  const auto &fe = system.get_fe();

  // Create mapping
  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());

  // Create quadrature
  hp::QCollection<dim> quadrature;
  quadrature.push_back(QGauss<dim>(fe_degree + 1));

  // Create constraints
  AffineConstraints<double> constraints;
  system.fill_constraints(constraints);
  constraints.close();

  // Categorize cells
  system.categorize();

  // Create sparsity pattern and allocate sparse matrix
  const unsigned int n_dofs = system.n_dofs();

  DynamicSparsityPattern dsp(n_dofs);
  system.create_sparsity_pattern(dsp);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  // create vectors
  Vector<double> rhs(n_dofs);
  Vector<double> solution(n_dofs);

  // compute matrix and right-hand side vector
  hp::FEValues<dim> fe_values_collection(mapping,
                                         fe,
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

      const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();

      // get indices
      dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      // compute element stiffness matrix
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) += fe_values.shape_grad(i, q_index) *
                                   fe_values.shape_grad(j, q_index) *
                                   fe_values.JxW(q_index);
        }

      // compute element vector
      Vector<double> cell_vector(dofs_per_cell);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          cell_vector(i) +=
            1.0 * fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);

      // assemble
      constraints.distribute_local_to_global(
        cell_matrix, cell_vector, dof_indices, sparse_matrix, rhs);
    }

  // solve problem
  ReductionControl         solver_control(100, 1.e-10, 1.e-4);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(sparse_matrix, solution, rhs, PreconditionIdentity());
  std::cout << solver_control.last_step() << std::endl << std::endl;

  // output result
  for (const auto &value : solution)
    std::cout << value << std::endl;

  // output result -> Paraview
  {
    FE_DGQ<dim>     fe_output(fe_degree_output);
    DoFHandler<dim> dof_handler_output(system.get_triangulation());
    dof_handler_output.distribute_dofs(fe_output);

    Vector<double> solution_output(dof_handler_output.n_dofs());

    hp::QCollection<dim> quadrature;
    quadrature.push_back(Quadrature<dim>(fe_output.get_unit_support_points()));

    hp::FEValues<dim> fe_values_collection(mapping,
                                           fe,
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

        const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();

        // get indices
        dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        // read vector
        Vector<double> cell_vector_input(dofs_per_cell);

        for (const unsigned int i : fe_values.dof_indices())
          cell_vector_input[i] = solution[dof_indices[i]];

        // perform interpolation
        Vector<double> cell_vector_output(fe_output.n_dofs_per_cell());

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            cell_vector_output(q_index) +=
              fe_values.shape_value(i, q_index) * cell_vector_input[i];

        // write
        cell->dealii_iterator()
          ->as_dof_handler_iterator(dof_handler_output)
          ->set_dof_values(cell_vector_output, solution_output);
      }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_output);
    data_out.add_data_vector(solution_output, "solution");
    data_out.build_patches(mapping, fe_degree_output);

    std::ofstream file("solution.vtu");
    data_out.write_vtu(file);
  }
}


int
main()
{
  test<1>();
}
