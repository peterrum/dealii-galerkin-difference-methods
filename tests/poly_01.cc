#include <deal.II/base/polynomial.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <deal.II/fe/fe_q_base.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

using namespace dealii;

template <int dim>
class FE_GDM : public FE_Q_Base<dim>
{
public:
  FE_GDM(const ScalarPolynomialsBase<dim> &poly)
    : FE_Q_Base<dim>(poly, create_data(poly.n()), std::vector<bool>(1, false))
  {}

  std::string
  get_name() const override
  {
    return "FE_GDM";
  }

  std::unique_ptr<FiniteElement<dim>>
  clone() const override
  {
    return std::make_unique<FE_GDM<dim>>(*this);
  }

private:
  static FiniteElementData<dim>
  create_data(const unsigned int n)
  {
    std::vector<unsigned int> dofs_per_object(dim + 1);
    dofs_per_object[dim] = n;

    FiniteElementData<dim> fe_data(dofs_per_object, 1, 0 /*not relevant*/);

    return fe_data;
  }
};


std::vector<Polynomials::Polynomial<double>>
generate_polynomials_1D()
{
  std::vector<std::vector<double>> coefficients = {
    {{{-1.0 / 6.0, 1.0 / 2.0, -1.0 / 3.0, 0.0}},
     {{1.0 / 2.0, -1.0, -1.0 / 2.0, 1.0}},
     {{-1.0 / 2.0, +1.0 / 2.0, 1.0, 0.0}},
     {{1.0 / 6.0, 0.0, -1. / 6., 0.0}}}};

  for (unsigned int i = 0; i < coefficients.size(); ++i)
    std::reverse(coefficients[i].begin(), coefficients[i].end());

  std::vector<Polynomials::Polynomial<double>> polynomials;
  for (unsigned int i = 0; i < coefficients.size(); ++i)
    polynomials.emplace_back(coefficients[i]);

  return polynomials;
}


int
main()
{
  const unsigned int dim            = 1;
  const unsigned int n_subdivisions = 20;

  const auto polynomials = generate_polynomials_1D();

  std::vector<std::vector<Polynomials::Polynomial<double>>> aniso_polynomials;
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

  FE_GDM<dim> fe(poly);

  MappingQ1<dim> mapping;
  QGauss<dim>    quad(polynomials.size());

  FEValues<dim> fe_values(mapping, fe, quad, update_values | update_JxW_values);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);

  fe_values.reinit(tria.begin());

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
            (fe_values.shape_value(i, q_index) *
             fe_values.shape_value(j, q_index) * fe_values.JxW(q_index));
    }

  cell_matrix.print_formatted(std::cout);
}
