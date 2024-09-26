#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <gdm/system.h>

using namespace dealii;

template <unsigned int dim>
struct Parameters
{
  // general settings
  std::string  simulation_type;
  unsigned int fe_degree;
  unsigned int n_components;

  // geometry
  unsigned int n_subdivisions_1D;
  double       geometry_left;
  double       geometry_right;

  // mass matrix
  double ghost_parameter_M = -1.0;

  // stiffness matrix
  double                         ghost_parameter_A = -1.0;
  double                         nitsche_parameter = -1.0;
  std::shared_ptr<Function<dim>> function_interface_dbc;
  std::shared_ptr<Function<dim>> function_rhs;

  // time stepping
  double start_t;
  double end_t;
  double cfl;
  double cfl_pow;

  // linear solver
  std::string  solver_name           = "AMG";
  unsigned int solver_max_iterations = 1000;
  double       solver_abs_tolerance  = 1.e-20;
  double       solver_rel_tolerance  = 1.e-14;

  // level set field
  unsigned int                   level_set_fe_degree;
  std::shared_ptr<Function<dim>> level_set_function;
};

template <unsigned int dim, typename Number>
class Discretization
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  Discretization() = default;

  void
  reinit(const Parameters<dim> &params)
  {
    // settings
    const unsigned int fe_degree           = params.fe_degree;
    const unsigned int n_subdivisions_1D   = params.n_subdivisions_1D;
    const unsigned int n_components        = params.n_components;
    const unsigned int level_set_fe_degree = params.level_set_fe_degree;
    const double       geometry_left       = params.geometry_left;
    const double       geometry_right      = params.geometry_left;
    const auto         level_set_function  = params.level_set_function;

    const MPI_Comm comm = MPI_COMM_WORLD;

    // Create GDM system
    system =
      std::make_shared<GDM::System<dim>>(comm, fe_degree, n_components, true);

    // Create mesh
    system->subdivided_hyper_cube(n_subdivisions_1D,
                                  geometry_left,
                                  geometry_right);

    dx = (geometry_right - geometry_left) / n_subdivisions_1D;

    // Create mapping
    mapping.push_back(MappingQ1<dim>());

    // Categorize cells
    system->categorize();

    const auto &tria = system->get_triangulation();

    // level set and classify cells
    level_set_dof_handler.reinit(tria);
    level_set_dof_handler.distribute_dofs(FE_Q<dim>(level_set_fe_degree));

    level_set.reinit(level_set_dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(
                       level_set_dof_handler),
                     comm);

    mesh_classifier =
      std::make_shared<NonMatching::MeshClassifier<dim>>(level_set_dof_handler,
                                                         level_set);

    VectorTools::interpolate(level_set_dof_handler,
                             *level_set_function,
                             level_set);

    level_set.update_ghost_values();
    mesh_classifier->reclassify();

    constraints.close();

    quadrature_1D   = QGauss<1>(fe_degree + 1);
    face_quadrature = QGauss<dim - 1>(fe_degree + 1);

    this->partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      system->locally_owned_dofs(),
      system->locally_relevant_dofs(constraints),
      comm);
  }

  const GDM::System<dim> &
  get_system() const
  {
    return *system;
  }

  const hp::FECollection<dim> &
  get_fe() const
  {
    return system->get_fe();
  }

  unsigned int
  get_fe_degree() const
  {
    return system->get_fe_degree();
  }

  const hp::MappingCollection<dim> &
  get_mapping() const
  {
    return mapping;
  }

  const Quadrature<1> &
  get_quadrature_1D() const
  {
    return quadrature_1D;
  }

  const Quadrature<dim - 1>
  get_face_quadrature() const
  {
    return face_quadrature;
  }

  const AffineConstraints<Number> &
  get_affine_constraints() const
  {
    return constraints;
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    vec.reinit(this->partitioner);
  }

  const DoFHandler<dim> &
  get_level_set_dof_handler() const
  {
    return level_set_dof_handler;
  }

  const VectorType &
  get_level_set() const
  {
    return level_set;
  }

  const NonMatching::MeshClassifier<dim> &
  get_mesh_classifier() const
  {
    return *mesh_classifier;
  }

  double
  get_dx() const
  {
    return dx;
  }

private:
  std::shared_ptr<GDM::System<dim>> system;
  hp::MappingCollection<dim>        mapping;
  Quadrature<1>                     quadrature_1D;
  Quadrature<dim - 1>               face_quadrature;
  AffineConstraints<Number>         constraints;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

  DoFHandler<dim>                                   level_set_dof_handler;
  VectorType                                        level_set;
  std::shared_ptr<NonMatching::MeshClassifier<dim>> mesh_classifier;

  double dx;
};
