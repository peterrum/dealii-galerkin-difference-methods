#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <gdm/system.h>

using namespace dealii;



template <unsigned int dim, typename Number>
class MassMatrixOperator
{
public:
  const TrilinosWrappers::SparseMatrix &
  get_sparse_matrix() const
  {
    compute_sparse_matrix();

    return sparse_matrix;
  }

private:
  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;

  void
  compute_sparse_matrix() const
  {
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    // settings
    const unsigned int fe_degree           = 3;
    const unsigned int n_subdivisions_1D   = 40;
    const unsigned int n_components        = 1;
    const unsigned int fe_degree_level_set = fe_degree;

    const double ghost_parameter_M = 0.25 * std::sqrt(3.0);

    const MPI_Comm comm = MPI_COMM_WORLD;

    // Create GDM system
    GDM::System<dim> system(comm, fe_degree, n_components, true);

    // Create mesh
    system.subdivided_hyper_cube(n_subdivisions_1D, -1.5, 1.5);

    // Create finite elements
    const auto &fe = system.get_fe();

    // Create mapping
    hp::MappingCollection<dim> mapping;
    mapping.push_back(MappingQ1<dim>());

    // Categorize cells
    system.categorize();

    const auto &tria = system.get_triangulation();

    // level set and classify cells
    const FE_Q<dim> fe_level_set(fe_degree_level_set);
    DoFHandler<dim> level_set_dof_handler(tria);
    level_set_dof_handler.distribute_dofs(fe_level_set);

    VectorType level_set;
    level_set.reinit(level_set_dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(
                       level_set_dof_handler),
                     comm);

    NonMatching::MeshClassifier<dim> mesh_classifier(level_set_dof_handler,
                                                     level_set);

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
                             level_set);

    level_set.update_ghost_values();
    mesh_classifier.reclassify();

    AffineConstraints<Number> constraints;
    constraints.close();

    const QGauss<1> quadrature_1D(fe_degree + 1);

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


    if (sparse_matrix.m() == 0 || sparse_matrix.n() == 0)
      {
        sparsity_pattern.reinit(system.locally_owned_dofs(), MPI_COMM_WORLD);
        system.create_flux_sparsity_pattern(constraints, sparsity_pattern);
        sparsity_pattern.compress();

        sparse_matrix.reinit(sparsity_pattern);
      }

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
      hp::QCollection<dim - 1>(QGauss<dim - 1>(fe_degree + 1)),
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
                      cell_matrix(i, j) += fe_values->shape_value(i, q_index) *
                                           fe_values->shape_value(j, q_index) *
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
                            .5 * ghost_parameter_M * cell_side_length *
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



template <typename Number>
class StiffnessMatrixOperator
{
public:
  const TrilinosWrappers::SparseMatrix &
  get_sparse_matrix() const
  {
    return sparse_matrix;
  }

private:
  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;
};



template <typename MatrixType>
void
compute_condition_number(const MatrixType &M_in)
{
  using Number = typename MatrixType::value_type;

  LAPACKFullMatrix<Number> M;

  M.copy_from(M_in);

  M.compute_eigenvalues();

  std::vector<Number> eigenvalues;
  for (unsigned int i = 0; i < M.m(); ++i)
    eigenvalues.push_back(M.eigenvalue(i).real());
  std::sort(eigenvalues.begin(), eigenvalues.end());

  std::cout << "condition number: " << eigenvalues.back() / eigenvalues.front()
            << std::endl;
}



template <typename MatrixType>
void
compute_max_generalized_eigenvalues_symmetric(const MatrixType &S_in,
                                              const MatrixType &M_in)
{
  using Number = typename MatrixType::value_type;

  LAPACKFullMatrix<Number> M;
  LAPACKFullMatrix<Number> S;

  M.copy_from(M_in);
  S.copy_from(S_in);

  std::vector<Vector<Number>> eigenvectors;
  std::vector<Number>         eigenvalues;

  S.compute_generalized_eigenvalues_symmetric(M, eigenvectors);

  for (unsigned int i = 0; i < S.m(); ++i)
    eigenvalues.push_back(S.eigenvalue(i).real());
  std::sort(eigenvalues.begin(), eigenvalues.end());


  std::cout << "max ev(M\\S):     " << eigenvalues.back() << std::endl;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  using Number           = double;
  const unsigned int dim = 1;

  MassMatrixOperator<dim, Number> mass_matrix_operator;
  StiffnessMatrixOperator<Number> stiffness_matrix_operator;

  if (true)
    compute_condition_number(mass_matrix_operator.get_sparse_matrix());

  if (false)
    compute_max_generalized_eigenvalues_symmetric(
      stiffness_matrix_operator.get_sparse_matrix(),
      mass_matrix_operator.get_sparse_matrix());
}