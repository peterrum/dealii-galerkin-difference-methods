#include <deal.II/base/mpi.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include "wave-discretization.h"
#include "wave-mass.h"
#include "wave-stiffness.h"

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

  std::cout << "condition number: " << eigenvalues.back() / eigenvalues.front()
            << std::endl;

  for (const auto i : eigenvalues)
    std::cout << i << std::endl;
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

  for (const auto i : eigenvalues)
    std::cout << i << std::endl;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  using Number           = double;
  const unsigned int dim = 1;

  Discretization<dim, Number> discretization;

  discretization.reinit();

  MassMatrixOperator<dim, Number>      mass_matrix_operator(discretization);
  StiffnessMatrixOperator<dim, Number> stiffness_matrix_operator(
    discretization);

  if (true)
    compute_condition_number(mass_matrix_operator.get_sparse_matrix());

  if (true)
    compute_max_generalized_eigenvalues_symmetric(
      stiffness_matrix_operator.get_sparse_matrix(),
      mass_matrix_operator.get_sparse_matrix());
}