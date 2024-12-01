#pragma once

#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <deal.II/fe/fe_q_base.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/hp/fe_collection.h>

using namespace dealii;

namespace GDM
{

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


  std::vector<std::vector<Polynomials::Polynomial<double>>>
  generate_polynomials_1D(const unsigned int fe_degree)
  {
    std::vector<std::vector<Polynomials::Polynomial<double>>> all_polynomial;

    // clang-format off
    std::vector<std::vector<std::vector<std::vector<double>>>> all_coefficients =
      {{
        // fe_degree == 1
        {{
          {{
            {{-1.0, +1.0}},
            {{+1.0, +0.0}}
          }}
        }},
        // fe_degree == 3
        {{
          {{
            {{-1.0 / 6.0, 1.0, -11.0 / 6.0, 1.0}},
            {{1.0 / 2.0, -5. / 2.0, 3.0, 0.0}},
            {{-1.0 / 2.0, +2.0, -3.0 / 2.0, 0.0}},
            {{1.0 / 6.0, -1. / 2., +1. / 3., 0.0}}
          }},
          {{
            {{-1.0 / 6.0, 1.0 / 2.0, -1.0 / 3.0, 0.0}},
            {{1.0 / 2.0, -1.0, -1.0 / 2.0, 1.0}},
            {{-1.0 / 2.0, +1.0 / 2.0, 1.0, 0.0}},
            {{1.0 / 6.0, 0.0, -1. / 6., 0.0}}
            }},
          {{
            {{-1.0 / 6.0, 0.0, 1.0 / 6.0, 0.0}},
            {{1.0 / 2.0, 1.0 / 2.0, -1.0, 0.0}},
            {{-1.0 / 2.0, -1.0, 1.0 / 2.0, 1.0}},
            {{1.0 / 6.0, 1.0 / 2.0, 1. / 3., 0.0}}
          }}
        }},
        // fe_degree == 5
        {{
          {{
            {{-1.0 / 120.0, 1.0 / 8.0, -17.0 / 24.0, 15.0 / 8.0, -137.0 / 60.0, 1.0 / 1.0}},
            {{1.0 / 24.0, -7.0 / 12.0, 71.0 / 24.0, -77.0 / 12.0, 5.0 / 1.0, 0.0 / 1.0}},
            {{-1.0 / 12.0, 13.0 / 12.0, -59.0 / 12.0, 107.0 / 12.0, -5.0 / 1.0, 0.0 / 1.0}},
            {{1.0 / 12.0, -1.0 / 1.0, 49.0 / 12.0, -13.0 / 2.0, 10.0 / 3.0, 0.0 / 1.0}},
            {{-1.0 / 24.0, 11.0 / 24.0, -41.0 / 24.0, 61.0 / 24.0, -5.0 / 4.0, 0.0 / 1.0}},
            {{1.0 / 120.0, -1.0 / 12.0, 7.0 / 24.0, -5.0 / 12.0, 1.0 / 5.0, 0.0 / 1.0}}
          }},
          {{
            {{-1.0 / 120.0, 1.0 / 12.0, -7.0 / 24.0, 5.0 / 12.0, -1.0 / 5.0, 0.0 / 1.0}},
            {{1.0 / 24.0, -3.0 / 8.0, 25.0 / 24.0, -5.0 / 8.0, -13.0 / 12.0, 1.0 / 1.0}},
            {{-1.0 / 12.0, 2.0 / 3.0, -17.0 / 12.0, -1.0 / 6.0, 2.0 / 1.0, 0.0 / 1.0}},
            {{1.0 / 12.0, -7.0 / 12.0, 11.0 / 12.0, 7.0 / 12.0, -1.0 / 1.0, 0.0 / 1.0}},
            {{-1.0 / 24.0, 1.0 / 4.0, -7.0 / 24.0, -1.0 / 4.0, 1.0 / 3.0, 0.0 / 1.0}},
            {{1.0 / 120.0, -1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, -1.0 / 20.0, 0.0 / 1.0}}
          }},
          {{
            {{-1.0 / 120.0, 1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0, 1.0 / 20.0, 0.0 / 1.0}},
            {{1.0 / 24.0, -1.0 / 6.0, -1.0 / 24.0, 2.0 / 3.0, -1.0 / 2.0, 0.0 / 1.0}},
            {{-1.0 / 12.0, 1.0 / 4.0, 5.0 / 12.0, -5.0 / 4.0, -1.0 / 3.0, 1.0 / 1.0}},
            {{1.0 / 12.0, -1.0 / 6.0, -7.0 / 12.0, 2.0 / 3.0, 1.0 / 1.0, 0.0 / 1.0}},
            {{-1.0 / 24.0, 1.0 / 24.0, 7.0 / 24.0, -1.0 / 24.0, -1.0 / 4.0, 0.0 / 1.0}},
            {{1.0 / 120.0, 0.0 / 1.0, -1.0 / 24.0, 0.0 / 1.0, 1.0 / 30.0, 0.0 / 1.0}}
          }},
          {{
            {{-1.0 / 120.0, 0.0 / 1.0, 1.0 / 24.0, 0.0 / 1.0, -1.0 / 30.0, 0.0 / 1.0}},
            {{1.0 / 24.0, 1.0 / 24.0, -7.0 / 24.0, -1.0 / 24.0, 1.0 / 4.0, 0.0 / 1.0}},
            {{-1.0 / 12.0, -1.0 / 6.0, 7.0 / 12.0, 2.0 / 3.0, -1.0 / 1.0, 0.0 / 1.0}},
            {{1.0 / 12.0, 1.0 / 4.0, -5.0 / 12.0, -5.0 / 4.0, 1.0 / 3.0, 1.0 / 1.0}},
            {{-1.0 / 24.0, -1.0 / 6.0, 1.0 / 24.0, 2.0 / 3.0, 1.0 / 2.0, 0.0 / 1.0}},
            {{1.0 / 120.0, 1.0 / 24.0, 1.0 / 24.0, -1.0 / 24.0, -1.0 / 20.0, 0.0 / 1.0}}
          }},
          {{
            {{-1.0 / 120.0, -1.0 / 24.0, -1.0 / 24.0, 1.0 / 24.0, 1.0 / 20.0, 0.0 / 1.0}},
            {{1.0 / 24.0, 1.0 / 4.0, 7.0 / 24.0, -1.0 / 4.0, -1.0 / 3.0, 0.0 / 1.0}},
            {{-1.0 / 12.0, -7.0 / 12.0, -11.0 / 12.0, 7.0 / 12.0, 1.0 / 1.0, 0.0 / 1.0}},
            {{1.0 / 12.0, 2.0 / 3.0, 17.0 / 12.0, -1.0 / 6.0, -2.0 / 1.0, 0.0 / 1.0}},
            {{-1.0 / 24.0, -3.0 / 8.0, -25.0 / 24.0, -5.0 / 8.0, 13.0 / 12.0, 1.0 / 1.0}},
            {{1.0 / 120.0, 1.0 / 12.0, 7.0 / 24.0, 5.0 / 12.0, 1.0 / 5.0, 0.0 / 1.0}}
          }}
        }}
      }};
    // clang-format on

    AssertIndexRange(fe_degree / 2, all_coefficients.size());

    for (auto coefficients : all_coefficients[fe_degree / 2])
      {
        for (unsigned int i = 0; i < coefficients.size(); ++i)
          std::reverse(coefficients[i].begin(), coefficients[i].end());

        std::vector<Polynomials::Polynomial<double>> polynomials;
        for (unsigned int i = 0; i < coefficients.size(); ++i)
          polynomials.emplace_back(coefficients[i]);

        all_polynomial.push_back(polynomials);
      }

    return all_polynomial;
  }


  template <int dim>
  std::array<unsigned int, dim>
  index_to_indices(const unsigned int                  index,
                   const std::array<unsigned int, dim> Ns)
  {
    std::array<unsigned int, dim> indices;

    if (dim >= 1)
      indices[0] = index % Ns[0];

    if (dim >= 2)
      indices[1] = (index / Ns[0]) % Ns[1];

    if (dim >= 3)
      indices[2] = index / (Ns[0] * Ns[1]);

    return indices;
  }


  template <int dim>
  std::array<unsigned int, dim>
  index_to_indices(const unsigned int index, const unsigned int N)
  {
    std::array<unsigned int, dim> Ns;
    std::fill(Ns.begin(), Ns.end(), N);
    return index_to_indices<dim>(index, Ns);
  }


  template <int dim>
  unsigned int
  indices_to_index(const std::array<unsigned int, dim> indices,
                   const std::array<unsigned int, dim> Ns)
  {
    unsigned int index = 0;

    if (dim >= 1)
      index += indices[0];

    if (dim >= 2)
      index += indices[1] * Ns[0];

    if (dim >= 3)
      index += indices[2] * Ns[0] * Ns[1];

    return index;
  }


  template <int dim>
  unsigned int
  indices_to_index(const std::array<unsigned int, dim> index,
                   const unsigned int                  N)
  {
    std::array<unsigned int, dim> Ns;
    std::fill(Ns.begin(), Ns.end(), N);
    return indices_to_index<dim>(index, Ns);
  }


  template <int dim>
  hp::FECollection<dim>
  generate_fe_collection(
    const std::vector<std::vector<Polynomials::Polynomial<double>>>
                      &all_polynomials_1D,
    const unsigned int n_components = 1)
  {
    hp::FECollection<dim> fe_collection;

    for (unsigned int p = 0; p < Utilities::pow(all_polynomials_1D.size(), dim);
         ++p)
      {
        std::vector<std::vector<Polynomials::Polynomial<double>>>
          aniso_polynomials;
        for (const auto d : index_to_indices<dim>(p, all_polynomials_1D.size()))
          aniso_polynomials.push_back(all_polynomials_1D[d]);

        fe_collection.push_back(FESystem<dim>(
          FE_GDM<dim>(AnisotropicPolynomials<dim>(aniso_polynomials)),
          n_components));
      }

    return fe_collection;
  }

} // namespace GDM
