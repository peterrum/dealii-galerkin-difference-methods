#include <deal.II/base/mpi.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include <gdm/advection/discretization.h>
#include <gdm/advection/mass.h>
#include <gdm/advection/stiffness.h>

#include <fstream>

using namespace dealii;

template <typename MatrixType>
void
compute_condition_number(const MatrixType &M_in,
                         const bool        rescale_matrix = false)
{
  using Number = typename MatrixType::value_type;

  LAPACKFullMatrix<Number> M;

  M.copy_from(M_in);

  if (rescale_matrix)
    {
      LAPACKFullMatrix<double> diagonal(M.m(), M.n());
      LAPACKFullMatrix<double> PA(M.m(), M.n());

      for (unsigned int i = 0; i < diagonal.m(); ++i)
        diagonal(i, i) = (M(i, i) == 0.0) ? 0.0 : (1.0 / M(i, i));

      diagonal.mmult(PA, M);

      M = PA;
    }

  M.compute_eigenvalues();

  std::vector<Number> eigenvalues;
  for (unsigned int i = 0; i < M.m(); ++i)
    eigenvalues.push_back(M.eigenvalue(i).real());
  std::sort(eigenvalues.begin(), eigenvalues.end());

  if (true)
    eigenvalues.erase(std::remove(eigenvalues.begin(), eigenvalues.end(), 1.0),
                      eigenvalues.end());

  std::cout << "condition number: " << eigenvalues.back() / eigenvalues.front()
            << std::endl;

  std::cout << "eigenvalues:" << std::endl;
  for (const auto i : eigenvalues)
    std::cout << i << " ";
  std::cout << std::endl;
  std::cout << std::endl;
}



template <typename MatrixType>
void
compute_max_generalized_eigenvalues_symmetric(const MatrixType &S_in,
                                              const MatrixType &M_in,
                                              const double      scaling)
{
  using Number = typename MatrixType::value_type;

  LAPACKFullMatrix<Number> M;
  LAPACKFullMatrix<Number> S;

  M.copy_from(M_in);
  M.invert();

  S.copy_from(S_in);

  LAPACKFullMatrix<Number> A(M.m(), M.n());
  M.mmult(A, S);

  std::vector<Vector<Number>>       eigenvectors;
  std::vector<std::complex<Number>> eigenvalues;

  A.compute_eigenvalues();

  for (unsigned int i = 0; i < A.m(); ++i)
    eigenvalues.push_back(A.eigenvalue(i));

  std::cout << "eigenvalues:" << std::endl;
  for (const auto i : eigenvalues)
    std::cout << (i.real() * scaling) << " " << (i.imag() * scaling)
              << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
}



template <typename MatrixType>
void
write_matrix_to_file(const MatrixType  &M_in,
                     const std::string &file_name,
                     const bool         write_binary_file)
{
  const auto flags =
    write_binary_file ? (std::ios::out | std::ios::binary) : (std::ios::out);
  ;

  std::ofstream file(file_name, flags);

  AssertThrow(file.is_open(), ExcInternalError());

  for (const auto &entry : M_in)
    {
      if (flags & std::ios::binary)
        {
          unsigned int row    = entry.row();
          unsigned int column = entry.column();
          double       value  = entry.value();

          file.write((char *)&row, sizeof(unsigned int));
          file.write((char *)&column, sizeof(unsigned int));
          file.write((char *)&value, sizeof(double));
        }
      else
        {
          file << entry.row() << " " << entry.column() << " " << entry.value()
               << std::endl;
        }
    }

  file.close();
}



struct MyParameters
{
  bool compute_kappa_M = false;
  bool compute_kappa_S = false;
  bool compute_gev     = false;

  bool        write_M           = false;
  bool        write_S           = false;
  std::string file_prefix       = "";
  bool        write_binary_file = true;
  bool        rescale_matrix    = false;
};



template <unsigned int dim>
void
parse_parameters(int              argc,
                 char           **argv,
                 Parameters<dim> &params,
                 MyParameters    &my_params)
{
  double       scale             = 1.0;
  unsigned int fe_degree         = 5;
  unsigned int n_subdivisions_1D = 100;

  double       alpha         = 1e-3;
  unsigned int n_ghost_cells = 0;

  for (int i = 1; i < argc;)
    {
      std::string label(argv[i]);

      if (label == "--disable_ghost_penalty")
        {
          scale = 0.0;
          i += 1;
        }
      else if (label == "--compute_kappa_m")
        {
          my_params.compute_kappa_M = true;
          i += 1;
        }
      else if (label == "--compute_gev")
        {
          my_params.compute_gev = true;
          i += 1;
        }
      else if (label == "--write_m")
        {
          my_params.write_M = true;
          i += 1;
        }
      else if (label == "--write_s")
        {
          my_params.write_S = true;
          i += 1;
        }
      else if (label == "--file_prefix")
        {
          my_params.file_prefix = std::string(argv[i + 1]);

          if (my_params.file_prefix != "")
            my_params.file_prefix += "_";

          i += 2;
        }
      else if (label == "--fe_degree")
        {
          fe_degree = std::atoi(argv[i + 1]);
          i += 2;
        }
      else if (label == "--n_subdivisions")
        {
          n_subdivisions_1D = std::atoi(argv[i + 1]);
          i += 2;
        }
      else if (label == "--n_ghost_cells")
        {
          n_ghost_cells = std::atoi(argv[i + 1]);
          i += 2;
        }
      else if (label == "--write_ascii")
        {
          my_params.write_binary_file = false;
          i += 1;
        }
      else if (label == "--rescale_matrix")
        {
          my_params.rescale_matrix = true;
          i += 1;
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

  if (n_ghost_cells == 0)
    {
      // gerneral settings
      params.fe_degree    = fe_degree;
      params.n_components = 1;
      params.composite    = false;

      params.solver_rel_tolerance = 1e-8;

      // geometry
      params.n_subdivisions_1D = n_subdivisions_1D;
      params.geometry_left     = -1.0;
      params.geometry_right    = +1.0;

      // mass matrix
      params.ghost_parameter_M = 0.5 * scale;

      // stiffness matrix
      params.ghost_parameter_A = 0.5;

      params.exact_solution =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            auto pp = p;
            pp[0] -= t;

            return std::exp(-16 * std::pow(pp.distance(Point<dim>(-0.3)), 2.0));
          });

      params.exact_solution_der =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) { return 0.0; });

      params.max_val = 1.0;

      params.advection = std::make_shared<FunctionFromFunctionObjects<dim>>(
        [](const auto &p, const unsigned int c) { return 1; }, dim);

      params.level_set_fe_degree = 1;
      params.level_set_function =
        std::make_shared<Functions::SignedDistance::Sphere<dim>>(Point<dim>{},
                                                                 1.2);
    }
  else
    {
      // gerneral settings
      params.fe_degree    = fe_degree;
      params.n_components = 1;
      params.composite    = false;

      params.solver_rel_tolerance = 1e-8;

      // geometry
      params.n_subdivisions_1D = n_subdivisions_1D + n_ghost_cells;
      params.geometry_left  = -1.0 - (2.0 / n_subdivisions_1D) * n_ghost_cells;
      params.geometry_right = +1.0;

      // mass matrix
      params.ghost_parameter_M = 0.5 * scale;

      // stiffness matrix
      params.ghost_parameter_A = 0.5 * scale;

      // time stepping
      params.start_t   = 0.0;
      params.end_t     = numbers::PI / 4.0;
      params.cfl       = 0.4;
      params.rk_method = RungeKuttaMethod::RK_AUTO;

      params.exact_solution =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) {
            auto pp = p;
            pp[0] -= t;

            return std::exp(-16 * std::pow(pp.distance(Point<dim>(-0.3)), 2.0));
          });

      params.exact_solution_der =
        std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
          [](const auto t, const auto &p) { return 0.0; });

      params.max_val = 1.0;

      params.advection = std::make_shared<FunctionFromFunctionObjects<dim>>(
        [](const auto &p, const unsigned int c) { return 1; }, dim);

      params.level_set_fe_degree = 1;

      const Point<dim> point(-1.0 - 2.0 / n_subdivisions_1D * alpha);
      Tensor<1, dim>   normal;
      normal[0] = -1;

      params.level_set_function =
        std::make_shared<Functions::SignedDistance::Plane<dim>>(point, normal);
    }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  using Number           = double;
  const unsigned int dim = 1;

  Parameters<dim> params;
  MyParameters    my_params;

  parse_parameters(argc, argv, params, my_params);

  Discretization<dim, Number>          discretization;
  MassMatrixOperator<dim, Number>      mass_matrix_operator(discretization);
  StiffnessMatrixOperator<dim, Number> stiffness_matrix_operator(
    discretization);

  discretization.reinit(params);
  mass_matrix_operator.reinit(params);
  stiffness_matrix_operator.reinit(params);

  if (my_params.compute_kappa_M)
    compute_condition_number(mass_matrix_operator.get_sparse_matrix(),
                             my_params.rescale_matrix);

  if (my_params.compute_gev)
    compute_max_generalized_eigenvalues_symmetric(
      stiffness_matrix_operator.get_sparse_matrix(),
      mass_matrix_operator.get_sparse_matrix(),
      discretization.get_dx() / params.max_val);

  if (my_params.write_M)
    write_matrix_to_file(mass_matrix_operator.get_sparse_matrix(),
                         my_params.file_prefix + "M.dat",
                         my_params.write_binary_file);

  if (my_params.write_S)
    write_matrix_to_file(stiffness_matrix_operator.get_sparse_matrix(),
                         my_params.file_prefix + "S.dat",
                         my_params.write_binary_file);
}
