#pragma once

#include <gdm/advection/parameters.h>

using namespace dealii;

template <unsigned int dim, typename Number>
class Discretization
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  Discretization() = default;

  void
  reinit(const Parameters<dim> &params)
  {
    (void)params;
  }
};
