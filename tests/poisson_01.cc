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
  AssertDimension(dim, 1);

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
  AssertDimension(dim, 1);

  const unsigned int cell_index = cell->active_cell_index();

  const int offset =
    (active_fe_index == 0) ? 0 : ((active_fe_index == 1) ? -1 : -2);

  for (unsigned int i = 0; i < fe[active_fe_index].n_dofs_per_cell(); ++i)
    {
      dof_indices[i] = cell_index + i + offset;
    }
}


int
main()
{
  const unsigned int dim              = 1;
  const unsigned int n_subdivisions   = 20;
  const unsigned int fe_degree        = 3;
  const unsigned int fe_degree_output = 2;

  // Create mesh
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions);

  // Create finite elements
  const auto all_polynomials = generate_polynomials_1D(fe_degree);
  const auto fe              = generate_fe_collection<dim>(all_polynomials);

  // Create mapping
  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());

  // Create quadrature
  hp::QCollection<dim> quadrature;
  quadrature.push_back(QGauss<dim>(fe_degree + 1));

  // Create constraints (TODO: generalize)
  AffineConstraints<double> constraints;
  constraints.constrain_dof_to_zero(0);
  constraints.constrain_dof_to_zero(n_subdivisions);
  constraints.close();

  // Categorize cells
  std::vector<unsigned int> active_fe_indices(tria.n_active_cells());

  for (const auto &cell : tria.active_cell_iterators())
    active_fe_indices[cell->active_cell_index()] = get_category(cell);

  // Create sparsity pattern and allocate sparse matrix
  DynamicSparsityPattern dsp(n_subdivisions + 1 /*TODO*/);

  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : tria.active_cell_iterators())
    {
      const auto active_fe_index = active_fe_indices[cell->active_cell_index()];
      dof_indices.resize(fe[active_fe_index].n_dofs_per_cell());
      get_dof_indices(dof_indices, cell, fe, active_fe_index);

      for (const auto i : dof_indices)
        dsp.add_entries(i, dof_indices.begin(), dof_indices.end());
    }

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  Vector<double> rhs(n_subdivisions + 1 /*TODO*/);
  Vector<double> solution(n_subdivisions + 1 /*TODO*/);

  // compute matrix and right-hand side vector
  hp::FEValues<dim> fe_values_collection(mapping,
                                         fe,
                                         quadrature,
                                         update_gradients | update_values |
                                           update_JxW_values);

  for (const auto &cell : tria.active_cell_iterators())
    {
      const auto active_fe_index = active_fe_indices[cell->active_cell_index()];

      fe_values_collection.reinit(cell,
                                  numbers::invalid_unsigned_int,
                                  numbers::invalid_unsigned_int,
                                  active_fe_index);

      const auto &fe_values = fe_values_collection.get_present_fe_values();

      const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();

      // get indices
      dof_indices.resize(dofs_per_cell);
      get_dof_indices(dof_indices, cell, fe, active_fe_index);

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
    DoFHandler<dim> dof_handler_output(tria);
    dof_handler_output.distribute_dofs(fe_output);

    Vector<double> solution_output(dof_handler_output.n_dofs());

    hp::QCollection<dim> quadrature;
    quadrature.push_back(Quadrature<dim>(fe_output.get_unit_support_points()));

    hp::FEValues<dim> fe_values_collection(mapping,
                                           fe,
                                           quadrature,
                                           update_gradients | update_values |
                                             update_JxW_values);

    for (const auto &cell : tria.active_cell_iterators())
      {
        const auto active_fe_index =
          active_fe_indices[cell->active_cell_index()];

        fe_values_collection.reinit(cell,
                                    numbers::invalid_unsigned_int,
                                    numbers::invalid_unsigned_int,
                                    active_fe_index);

        const auto &fe_values = fe_values_collection.get_present_fe_values();

        const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();

        // get indices
        dof_indices.resize(dofs_per_cell);
        get_dof_indices(dof_indices, cell, fe, active_fe_index);

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
        cell->as_dof_handler_iterator(dof_handler_output)
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
