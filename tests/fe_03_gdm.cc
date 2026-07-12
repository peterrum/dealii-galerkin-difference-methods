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

  Point<dim> point0(0.0);
  Point<dim> point1(1.0);

  for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    {
      my_printf(fe.shape_grad(i, point0)[0]);
      my_printf(fe.shape_grad(i, point1)[0]);
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

  for (const auto &fe : fe_collection)
    test(fe);
}


int
main()
{
  test<1>(5);
}
