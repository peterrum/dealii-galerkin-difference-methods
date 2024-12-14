#include <deal.II/base/quadrature_lib.h>

#include <deal.II/hp/fe_collection.h>

#include <gdm/fe.h>

#include <fstream>

using namespace dealii;

template <int dim>
void
test(const unsigned int fe_degree, const unsigned int n_components = 1)
{
  const unsigned int n_subdivisions = 10;

  const QIterated<dim> quadrature(QGaussLobatto<1>(2), n_subdivisions);

  const auto fe_collection =
    GDM::generate_fe_collection<dim>(GDM::generate_polynomials_1D(fe_degree),
                                     n_components);

  for (const auto &fe : fe_collection)
    {
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            printf("%7.3f ", fe.shape_value(i, quadrature.point(q)));
          std::cout << std::endl;
        }
      std::cout << std::endl;
    }
  std::cout << std::endl << std::endl << std::endl;
}


int
main()
{
  test<1>(3);
  test<2>(3);
  test<3>(3);

  test<1>(5);
  test<1>(7);
  test<1>(9);

  test<1>(3, 2);
}
