#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_product_polynomials.h>

using namespace dealii;

int
main()
{
  const unsigned int dim            = 1;
  const unsigned int n_subdivisions = 20;

  std::vector<std::vector<double>> coefficients = {
    {{{0., -1. / 6., 0., 1. / 6.}},
     {{0., 1., 1. / 2., -1. / 2.}},
     {{1., -1. / 2., -1., 1. / 2.}},
     {{0., -1. / 3., 1. / 2., -1. / 6.}}}};


  std::vector<Polynomials::Polynomial<double>> polynomials;
  for (unsigned int i = 0; i < coefficients.size(); ++i)
    polynomials.emplace_back(coefficients[i]);

  std::vector<std::vector<Polynomials::Polynomial<double>>> aniso_polynomials;
  for (unsigned int i = 0; i < dim; ++i)
    aniso_polynomials.push_back(polynomials);

  AnisotropicPolynomials<dim> poly(aniso_polynomials);

  for (unsigned int j = 0; j <= n_subdivisions; ++j)
    {
      Point<dim> x;
      x[0] = 1.0 / n_subdivisions * j;

      for (unsigned int i = 0; i < coefficients.size(); ++i)
        printf("%7.3f ", poly.compute_value(i, x));
      std::cout << std::endl;
    }
}
