#pragma once

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <gdm/advection/discretization.h>

using namespace dealii;

template <unsigned int dim, typename Number>
class StiffnessMatrixOperator
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  StiffnessMatrixOperator(const Discretization<dim, Number> &discretization)
    : discretization(discretization)
    , ghost_parameter_A(-1.0)
  {}

  void
  reinit(const Parameters<dim> &params)
  {
    this->ghost_parameter_A = params.ghost_parameter_A;
  }

  void
  compute_rhs(BlockVectorType       &vec_rhs,
              const BlockVectorType &solution,
              const bool             compute_impl_part,
              const double           time) const
  {
    /*TODO*/
  }

private:
  const Discretization<dim, Number> &discretization;

  double ghost_parameter_A;

  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;

  void
  compute_sparse_matrix() const
  {
    // TODO
  }
};
