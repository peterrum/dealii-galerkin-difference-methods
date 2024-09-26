#include <deal.II/base/mpi.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include <gdm/wave/wave-discretization.h>
#include <gdm/wave/wave-mass.h>
#include <gdm/wave/wave-stiffness.h>

using namespace dealii;

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

  if (true)
    eigenvalues.erase(std::remove(eigenvalues.begin(), eigenvalues.end(), 1.0),
                      eigenvalues.end());

  std::cout << "condition number: " << eigenvalues.back() / eigenvalues.front()
            << std::endl;

  std::cout << "eigenvalues:" << std::endl;
  for (const auto i : eigenvalues)
    std::cout << i << " ";
  std::cout << std::endl;
  std::cout << std::endl;
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

  std::cout << "max ev(M\\S): " << eigenvalues.back() << std::endl;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  using Number           = double;
  const unsigned int dim = 1;

  Parameters<dim> params;

  // general settings
  params.fe_degree    = 3;
  params.n_components = 1;

  // geometry
  params.n_subdivisions_1D = 40;
  params.geometry_left     = -1.21;
  params.geometry_right    = +1.21;

  // mass matrix
  params.ghost_parameter_M = 0.25 * std::sqrt(3.0);

  // stiffness matrix
  params.ghost_parameter_A      = 0.50 * std::sqrt(3.0);
  params.nitsche_parameter      = 5.0 * params.fe_degree;
  params.function_interface_dbc = {};
  params.function_rhs           = {};

  // level set field
  params.level_set_fe_degree = params.fe_degree;
  params.level_set_function =
    std::make_shared<Functions::SignedDistance::Sphere<dim>>();

  Discretization<dim, Number>          discretization;
  MassMatrixOperator<dim, Number>      mass_matrix_operator(discretization);
  StiffnessMatrixOperator<dim, Number> stiffness_matrix_operator(
    discretization);

  discretization.reinit(params);
  mass_matrix_operator.reinit(params);
  stiffness_matrix_operator.reinit(params);

  if (true)
    compute_condition_number(mass_matrix_operator.get_sparse_matrix());

  if (true)
    compute_max_generalized_eigenvalues_symmetric(
      stiffness_matrix_operator.get_sparse_matrix(),
      mass_matrix_operator.get_sparse_matrix());
}