#include <gdm/fe.h>

using namespace dealii;


int
main()
{
  const unsigned int dim            = 1;
  const unsigned int n_subdivisions = 20;
  const unsigned int fe_degree      = 3;

  const auto all_polynomials = GDM::generate_polynomials_1D(fe_degree);

  for (const auto &polynomials : all_polynomials)
    {
      std::vector<std::vector<Polynomials::Polynomial<double>>>
        aniso_polynomials;
      for (unsigned int i = 0; i < dim; ++i)
        aniso_polynomials.push_back(polynomials);

      AnisotropicPolynomials<dim> poly(aniso_polynomials);

      for (unsigned int j = 0; j <= n_subdivisions; ++j)
        {
          Point<dim> x;
          x[0] = 1.0 / n_subdivisions * j;

          for (unsigned int i = 0; i < polynomials.size(); ++i)
            printf("%7.3f ", poly.compute_value(i, x));
          std::cout << std::endl;
        }
      std::cout << std::endl << std::endl;
    }
}
