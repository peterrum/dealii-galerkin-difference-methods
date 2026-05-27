#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <gdm/system.h>
#include <gdm/wave/parameters.h>

using namespace dealii;

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
    const double       geometry_right      = params.geometry_right;
    const auto         level_set_functions = params.level_set_functions;

    const MPI_Comm comm = MPI_COMM_WORLD;

    // Create GDM system
    system =
      std::make_shared<GDM::System<dim>>(comm, fe_degree, n_components, true);

    // Create mesh
    system->subdivided_hyper_cube(n_subdivisions_1D,
                                  geometry_left,
                                  geometry_right);

    const auto &tria = system->get_triangulation();

    dx = (geometry_right - geometry_left) / n_subdivisions_1D;

    // Create mapping
    if (params.mapping_q_cache_function)
      {
        MappingQ1<dim> mapping_q1;

        MappingQCache<dim> mapping_q_cache(1 /*TODO*/);

        mapping_q_cache.initialize(
          mapping_q1,
          tria,
          [&](const auto &, const auto &p) {
            return params.mapping_q_cache_function(p);
          },
          false);

        mapping.push_back(mapping_q_cache);
      }
    else
      mapping.push_back(MappingQ1<dim>());

    // Categorize cells
    system->categorize();

    // level set and classify cells
    level_set_dof_handler.reinit(tria);
    level_set_dof_handler.distribute_dofs(FE_Q<dim>(level_set_fe_degree));

    if (level_set_functions.size() == 1)
      {
        level_sets.resize(2);
        for (unsigned int i = 0; i < 2; ++i)
          {
            level_sets[i].reinit(level_set_dof_handler.locally_owned_dofs(),
                                 DoFTools::extract_locally_relevant_dofs(
                                   level_set_dof_handler),
                                 comm);
            VectorTools::interpolate(level_set_dof_handler,
                                     *level_set_functions[0],
                                     level_sets[i]);
            if (i == 1)
              level_sets[i] *= -1.0;
            level_sets[i].update_ghost_values();

            mesh_classifiers.push_back(
              std::make_shared<NonMatching::MeshClassifier<dim>>(
                level_set_dof_handler, level_sets[i]));
            mesh_classifiers[i]->reclassify();
          }
      }
    else
      {
        level_sets.resize(level_set_functions.size());
        for (unsigned int i = 0; i < level_set_functions.size(); ++i)
          {
            level_sets[i].reinit(level_set_dof_handler.locally_owned_dofs(),
                                 DoFTools::extract_locally_relevant_dofs(
                                   level_set_dof_handler),
                                 comm);
            VectorTools::interpolate(level_set_dof_handler,
                                     *level_set_functions[i],
                                     level_sets[i]);
            level_sets[i].update_ghost_values();

            mesh_classifiers.push_back(
              std::make_shared<NonMatching::MeshClassifier<dim>>(
                level_set_dof_handler, level_sets[i]));
            mesh_classifiers[i]->reclassify();
          }
      }


    quadrature_1D   = QGauss<1>(fe_degree + 1);
    face_quadrature = QGauss<dim - 1>(fe_degree + 1);

    // ── Constraints ──────────────────────────────────────────────
    const int n_domains = level_sets.size();
    // constraints.resize(n_domains);
    constraints.close();
    partitioners.resize(n_domains);

    for (int i = 0; i < n_domains; ++i)
      {
        // constraints[i].close();
        partitioners[i] = std::make_shared<const Utilities::MPI::Partitioner>(
          system->locally_owned_dofs(),
          system->locally_relevant_dofs(constraints),
          comm);
      }
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
  initialize_dof_vector(VectorType &vec, const int &i) const
  {
    vec.reinit(partitioners[i]);
  }

  const DoFHandler<dim> &
  get_level_set_dof_handler() const
  {
    return level_set_dof_handler;
  }

  const std::vector<VectorType> &
  get_level_sets() const
  {
    return level_sets;
  }

  const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> &
  get_mesh_classifiers() const
  {
    return mesh_classifiers;
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

  DoFHandler<dim>           level_set_dof_handler;
  AffineConstraints<Number> constraints;

  std::vector<VectorType> level_sets;
  std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>>
    mesh_classifiers;
  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;

  double dx;
};
