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
class Parameters
{};

template <unsigned int dim, typename Number>
class Discretization
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  Discretization() = default;

  void
  reinit()
  {
    // settings
    const unsigned int fe_degree           = 5;
    const unsigned int n_subdivisions_1D   = 40;
    const unsigned int n_components        = 1;
    const unsigned int fe_degree_level_set = fe_degree;

    const MPI_Comm comm = MPI_COMM_WORLD;

    // Create GDM system
    system =
      std::make_shared<GDM::System<dim>>(comm, fe_degree, n_components, true);

    // Create mesh
    system->subdivided_hyper_cube(n_subdivisions_1D, -1.5, 1.5);

    // Create mapping
    mapping.push_back(MappingQ1<dim>());

    // Categorize cells
    system->categorize();

    const auto &tria = system->get_triangulation();

    // level set and classify cells
    level_set_dof_handler.reinit(tria);
    level_set_dof_handler.distribute_dofs(FE_Q<dim>(fe_degree_level_set));

    level_set.reinit(level_set_dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(
                       level_set_dof_handler),
                     comm);

    mesh_classifier =
      std::make_shared<NonMatching::MeshClassifier<dim>>(level_set_dof_handler,
                                                         level_set);

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
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
};
