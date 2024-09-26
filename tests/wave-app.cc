#include <deal.II/base/mpi.h>

#include "wave-discretization.h"
#include "wave-mass.h"
#include "wave-stiffness.h"


using namespace dealii;

template <int dim, typename Number = double>
void
run()
{
  Discretization<dim, Number> discretization;

  MassMatrixOperator<dim, Number>      mass_matrix_operator(discretization);
  StiffnessMatrixOperator<dim, Number> stiffness_matrix_operator(
    discretization);

  discretization.reinit();

  const std::string simulation_type = "poisson";

  if (simulation_type == "poisson")
    {
      AssertThrow(false, ExcNotImplemented());

      // Compute stiffness matrix

      // Compute right-hand-side vector

      // setup solver

      // solve

      // postprocess
    }
  else if (simulation_type == "heat-rk")
    {
      AssertThrow(false, ExcNotImplemented());
    }
  else if (simulation_type == "heat-impl")
    {
      AssertThrow(false, ExcNotImplemented());
    }
  else if (simulation_type == "wave-rk")
    {
      AssertThrow(false, ExcNotImplemented());
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim = 2;

  if (dim == 1)
    run<1>();
  else if (dim == 2)
    run<2>();
  else
    AssertThrow(false, ExcNotImplemented());
}
