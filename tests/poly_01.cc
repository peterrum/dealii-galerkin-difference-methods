#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gdm/fe.h>

using namespace dealii;


int
main()
{
  const unsigned int dim            = 1;
  const unsigned int n_subdivisions = 20;
  const unsigned int fe_degree      = 3;

  const auto all_polynomials = generate_polynomials_1D(fe_degree);
  const auto fe_collection   = generate_fe_collection<dim>(all_polynomials);

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

      continue;

      FE_GDM<dim> fe(poly);

      MappingQ1<dim> mapping;
      QGauss<dim>    quad(polynomials.size());

      FEValues<dim> fe_values(mapping,
                              fe,
                              quad,
                              update_values | update_JxW_values);

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
}
