#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/hp/fe_collection.h>

#include <gdm/fe.h>

#include <fstream>

using namespace dealii;

template <typename Number>
void
my_printf(const Number &value)
{
  if (value >= 0.0)
    printf("%10.3f ", value);
  else
    printf("%10.3f ", std::abs(value));
}

template <int dim>
void
test(const FiniteElement<dim> &fe)
{
  std::cout << fe.get_name() << ":" << std::endl;

  Point<dim> point;

  for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    {
      my_printf(fe.shape_value(i, point));
      my_printf(fe.shape_grad(i, point)[0]);
      my_printf(fe.shape_grad_grad(i, point)[0][0]);
      my_printf(fe.shape_3rd_derivative(i, point)[0][0][0]);
      my_printf(fe.shape_4th_derivative(i, point)[0][0][0][0]);
      std::cout << std::endl;
    }

  std::cout << std::endl << std::endl << std::endl;
}

template <int dim>
void
test(const unsigned int fe_degree)
{
  const auto fe_collection =
    GDM::generate_fe_collection<dim>(GDM::generate_polynomials_1D(fe_degree));

  test(fe_collection[fe_degree / 2]);

  test(FE_DGQ<dim>(fe_degree));
}


int
main()
{
  test<1>(3);
  test<1>(5);
}
