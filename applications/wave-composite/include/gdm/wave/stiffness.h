#pragma once

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <gdm/wave/discretization.h>

using namespace dealii;

template <unsigned int dim, typename Number>
class StiffnessMatrixOperator
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  StiffnessMatrixOperator(const Discretization<dim, Number> &discretization)
    : discretization(discretization)
    , ghost_parameter_A(-1.0)
    , nitsche_parameter(-1.0)
  {}

  void
  reinit(const Parameters<dim> &params)
  {
    this->ghost_parameter_A = params.ghost_parameter_A;
    this->nitsche_parameter = params.nitsche_parameter;

    this->function_domain_dbc    = params.function_domain_dbc;
    this->function_interface_dbc = params.function_interface_dbc;
    this->function_rhs           = params.function_rhs;
    this->speed                  = params.speed;
  }

  const std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &
  get_sparse_matrix() const
  {
    block_sparse_matrix.clear();
    block_sparse_matrix.resize(discretization.get_level_sets().size());
    for(size_t i = 0; i < discretization.get_level_sets().size(); ++i){
      block_sparse_matrix[i] = std::make_shared<TrilinosWrappers::SparseMatrix>();
      compute_sparse_matrix(i, *block_sparse_matrix[i]);
    }
    return block_sparse_matrix;
  }

  void
  compute_rhs_internal(size_t domain_idx, 
                      VectorType                           &vec_rhs,
                       const VectorType                     &solution,
                       const bool                            compute_impl_part,
                       const double                          time) const
  {
    std::shared_ptr<Function<dim>> required_speed = speed[domain_idx];
    
    // 0) extract information from discretization class
    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const Quadrature<1> &quadrature_1D = discretization.get_quadrature_1D();
    const Quadrature<dim - 1> &face_quadrature =
      discretization.get_face_quadrature();
    const GDM::System<dim>          &system = discretization.get_system();
    const AffineConstraints<Number> &constraints =
      discretization.get_affine_constraints();
    const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> &mesh_classifiers =
      discretization.get_mesh_classifiers();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const std::vector<VectorType>              &level_sets = discretization.get_level_sets();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    AssertThrow(ghost_parameter_A != -1.0, ExcNotImplemented());

    if (function_interface_dbc)
      function_interface_dbc->set_time(time);

    if (function_domain_dbc)
      function_domain_dbc->set_time(time);

    if (function_rhs)
      function_rhs->set_time(time);

    const auto face_has_ghost_penalty = [&](const auto        &cell,
                                            const unsigned int face_index) {
      if (cell->at_boundary(face_index))
        return false;

      const NonMatching::LocationToLevelSet cell_location =
        mesh_classifiers[domain_idx]->location_to_level_set(cell);

        const NonMatching::LocationToLevelSet neighbor_location =
          mesh_classifiers[domain_idx]->location_to_level_set(cell->neighbor(face_index));

        if (cell_location == NonMatching::LocationToLevelSet::intersected &&
            neighbor_location != NonMatching::LocationToLevelSet::outside)
          return true;

        if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
            cell_location != NonMatching::LocationToLevelSet::outside)
          return true;

        return false;
    };

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                   update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      *mesh_classifiers[domain_idx],
                                                      level_set_dof_handler,
                                                      level_sets[domain_idx]);

    NonMatching::RegionUpdateFlags region_update_flags_face;
    region_update_flags_face.inside =
        update_values | update_gradients | update_JxW_values |
        update_quadrature_points | update_normal_vectors;

    NonMatching::FEInterfaceValues<dim> non_matching_fe_interface_values(
      fe,
      quadrature_1D,
      region_update_flags_face,
      *mesh_classifiers[domain_idx],
      level_set_dof_handler,
      level_sets[domain_idx]);  

    FEInterfaceValues<dim> fe_interface_values(
      mapping,
      fe,
      hp::QCollection<dim - 1>(face_quadrature),
      update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points);

    solution.update_ghost_values();

    for (const auto &cell : system.locally_active_cell_iterators())
      if (cell->is_locally_owned() &&
          (mesh_classifiers[domain_idx]->location_to_level_set(cell->dealii_iterator()) !=
           NonMatching::LocationToLevelSet::outside))
        {
          non_matching_fe_values.reinit(cell->dealii_iterator(),
                                        numbers::invalid_unsigned_int,
                                        numbers::invalid_unsigned_int,
                                        cell->active_fe_index());

          const double cell_side_length =
            cell->dealii_iterator()->minimum_vertex_distance();

          const unsigned int n_dofs_per_cell = fe[0].dofs_per_cell;

          std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          Vector<Number> cell_vector(n_dofs_per_cell);

          // (I) cell integral
          if (const auto &fe_values_ptr = non_matching_fe_values.get_inside_fe_values())
            {
              const auto &fe_values = *fe_values_ptr;

              std::vector<Tensor<1, dim, Number>> quadrature_gradients(
                fe_values.n_quadrature_points);
              fe_values.get_function_gradients(solution,
                                               dof_indices,
                                               quadrature_gradients);

              for (const unsigned int q : fe_values.quadrature_point_indices())
                {
                  const Point<dim> &point = fe_values.quadrature_point(q);

                  double speed_cell= required_speed->value(point);

                  for (const unsigned int i : fe_values.dof_indices())
                    {
                      // left hand side: (∇v, ∇u)
                      if (compute_impl_part)
                        cell_vector(i) -= speed_cell * fe_values.shape_grad(i, q) *
                                          quadrature_gradients[q] *
                                          fe_values.JxW(q);

                      // right hand side: (v, f)
                      if (function_rhs)
                        cell_vector(i) += function_rhs->value(point) *
                                          fe_values.shape_value(i, q) *
                                          fe_values.JxW(q);
                    }
                }
            }

          // (II) surface integral to apply BC
          if (function_interface_dbc)
            if (const auto &surface_fe_values_ptr =
                  non_matching_fe_values.get_surface_fe_values())
              {
                const auto &surface_fe_values = *surface_fe_values_ptr;

                std::vector<Number> quadrature_values(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_values(solution,
                                                      dof_indices,
                                                      quadrature_values);

                std::vector<Tensor<1, dim, Number>> quadrature_gradients(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_gradients(solution,
                                                         dof_indices,
                                                         quadrature_gradients);


                for (const unsigned int q :
                     surface_fe_values.quadrature_point_indices())
                  {
                    const Point<dim> &point =
                      surface_fe_values.quadrature_point(q);
                    const Tensor<1, dim> normal =
                      surface_fe_values.normal_vector(q);
                    
                    double c_surface= required_speed->value(point);

                    for (const unsigned int i : surface_fe_values.dof_indices())
                      {
                        // left hand side:
                        // - <v, ∂u/∂n> - <∂v/∂n, u> + γ_D/h <v, u>
                        if (compute_impl_part)
                          cell_vector(i) -=
                            (-normal * surface_fe_values.shape_grad(i, q) *
                               quadrature_values[q] +
                             -normal * quadrature_gradients[q] *
                               surface_fe_values.shape_value(i, q) +
                             nitsche_parameter / cell_side_length *
                               surface_fe_values.shape_value(i, q) *
                               quadrature_values[q]) * c_surface *
                            surface_fe_values.JxW(q);

                        // right hand side: <γ_D/h v - ∂v/∂n, g_D>
                        cell_vector(i) +=
                          function_interface_dbc->value(point) * c_surface *
                          (nitsche_parameter / cell_side_length *
                             surface_fe_values.shape_value(i, q) -
                           normal * surface_fe_values.shape_grad(i, q)) *
                          surface_fe_values.JxW(q);
                      }
                  }
              }

          // (IV) face integral for apply DBC
          if (function_domain_dbc)
            for (const auto f : cell->dealii_iterator()->face_indices())
              if (cell->dealii_iterator()->face(f)->at_boundary())
                {
                  non_matching_fe_interface_values.reinit(
                    cell->dealii_iterator(),
                    f,
                    numbers::invalid_unsigned_int,
                    numbers::invalid_unsigned_int,
                    cell->active_fe_index());

                  if (const auto &surface_fe_values_ptr = non_matching_fe_interface_values.get_inside_fe_values())
                    {
                      const auto &surface_fe_values =
                        surface_fe_values_ptr->get_fe_face_values(0);

                      std::vector<Number> quadrature_values(
                        surface_fe_values.n_quadrature_points);
                      surface_fe_values.get_function_values(solution,
                                                            dof_indices,
                                                            quadrature_values);

                      std::vector<Tensor<1, dim, Number>> quadrature_gradients(
                        surface_fe_values.n_quadrature_points);
                      surface_fe_values.get_function_gradients(
                        solution, dof_indices, quadrature_gradients);


                      for (const unsigned int q :
                           surface_fe_values.quadrature_point_indices())
                        {
                          const Point<dim> &point =
                            surface_fe_values.quadrature_point(q);
                          const Tensor<1, dim> normal =
                            surface_fe_values.normal_vector(q);
                          double c_surface = required_speed->value(point);

                          for (const unsigned int i :
                               surface_fe_values.dof_indices())
                            {
                              // left hand side:
                              // - <v, ∂u/∂n> - <∂v/∂n, u> + γ_D/h <v, u>
                              if (compute_impl_part)
                                cell_vector(i) -=
                                  (-normal *
                                     surface_fe_values.shape_grad(i, q) *
                                     quadrature_values[q] +
                                   -normal * quadrature_gradients[q] *
                                     surface_fe_values.shape_value(i, q) +
                                   nitsche_parameter / cell_side_length *
                                     surface_fe_values.shape_value(i, q) *
                                     quadrature_values[q]) * c_surface *
                                  surface_fe_values.JxW(q);

                              // right hand side: <γ_D/h v - ∂v/∂n, g_D>
                              cell_vector(i) +=
                                function_domain_dbc->value(point) * c_surface *
                                (nitsche_parameter / cell_side_length *
                                   surface_fe_values.shape_value(i, q) -
                                 normal * surface_fe_values.shape_grad(i, q)) *
                                surface_fe_values.JxW(q);
                            }
                        }
                    }
                }

          // (V) face integral for apply GP
          if (compute_impl_part)
            for (const unsigned int f : cell->dealii_iterator()->face_indices())
              if (face_has_ghost_penalty(cell->dealii_iterator(), f))
                {
                  fe_interface_values.reinit(
                    cell->dealii_iterator(),
                    f,
                    numbers::invalid_unsigned_int,
                    cell->dealii_iterator()->neighbor(f),
                    cell->dealii_iterator()->neighbor_of_neighbor(f),
                    numbers::invalid_unsigned_int,
                    numbers::invalid_unsigned_int,
                    numbers::invalid_unsigned_int,
                    cell->active_fe_index(),
                    cell->neighbor(f)->active_fe_index());

                  const unsigned int n_interface_dofs =
                    fe_interface_values.n_current_interface_dofs();
                  Vector<double> local_stabilization(n_interface_dofs);

                  std::vector<types::global_dof_index>
                    local_interface_dof_indices;
                  cell->get_dof_indices(dof_indices);
                  for (const auto i : dof_indices)
                    local_interface_dof_indices.emplace_back(i);
                  cell->neighbor(f)->get_dof_indices(dof_indices);
                  for (const auto i : dof_indices)
                    local_interface_dof_indices.emplace_back(i);

                  std::vector<Tensor<1, dim>> jump_in_shape_gradients(
                    fe_interface_values.n_quadrature_points);

                  const FEValuesExtractors::Scalar scalar(0);

                  std::vector<double> local_dof_values(n_interface_dofs);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    local_dof_values[i] =
                      solution[local_interface_dof_indices[i]];

                  fe_interface_values[scalar]
                    .get_jump_in_function_gradients_from_local_dof_values(
                      local_dof_values, jump_in_shape_gradients);

                  for (unsigned int q = 0;
                       q < fe_interface_values.n_quadrature_points;
                       ++q)
                    {
                      const Tensor<1, dim> normal =
                        fe_interface_values.normal(q);
                      const Point<dim> point= fe_interface_values.quadrature_point(q);
                      double c_interface= required_speed->value(point);
                      for (unsigned int i = 0; i < n_interface_dofs; ++i)
                        {
                          // γ_A j(v, u) / h^2 with j(v, u)= ∑ h^3 <∂v/∂n,
                          // ∂u/∂n>
                          local_stabilization(i) -=
                            .5 * ghost_parameter_A * c_interface * cell_side_length *
                            (normal *
                             fe_interface_values.jump_in_shape_gradients(i,q)) *
                            (normal * jump_in_shape_gradients[q]) *
                            fe_interface_values.JxW(q);
                        }
                    }

                  constraints.distribute_local_to_global(
                    local_stabilization, local_interface_dof_indices, vec_rhs);
                }

          cell->get_dof_indices(dof_indices);
          constraints.distribute_local_to_global(cell_vector,
                                                 dof_indices,
                                                 vec_rhs);
        }

    vec_rhs.compress(VectorOperation::add);
  }

  void
  compute_rhs(const size_t domain_idx,
              VectorType       &vec_rhs,
              const VectorType &solution,
              const bool        compute_impl_part,
              const double      time) const
  {
    compute_rhs_internal(domain_idx,
                         vec_rhs,
                         solution,
                         compute_impl_part,
                         time);
  }

  void
  compute_rhs(BlockVectorType       &vec_rhs,
              const BlockVectorType &solution,
              const bool             compute_impl_part,
              const double           time) const
  {
    AssertThrow(compute_impl_part, ExcNotImplemented());

    for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size(); ++domain_idx){
      compute_rhs_internal(domain_idx, vec_rhs.block(domain_idx), solution.block(domain_idx), true, time);    
    }

    // add coupling term
    if (function_interface_dbc)
      return; // nothing to do

    // 0) extract information from discretization class
    const Quadrature<1> &quadrature_1D = discretization.get_quadrature_1D();
    const Quadrature<dim - 1> &face_quadrature =
      discretization.get_face_quadrature();
    const GDM::System<dim>          &system = discretization.get_system();
    const AffineConstraints<Number> &constraints =
      discretization.get_affine_constraints();
    const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> &mesh_classifiers =
      discretization.get_mesh_classifiers();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const std::vector<VectorType>              &level_sets = discretization.get_level_sets();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    
    for(size_t domain_idx = 0; domain_idx < discretization.get_level_sets().size() - 1; ++domain_idx){
      NonMatching::RegionUpdateFlags region_update_flags;
      region_update_flags.surface = update_values | update_gradients |
                                    update_JxW_values | update_quadrature_points |
                                    update_normal_vectors;

      NonMatching::FEValues<dim> non_matching_fe_values(fe,
                                                        quadrature_1D,
                                                        region_update_flags,
                                                        *mesh_classifiers[domain_idx],
                                                          level_set_dof_handler,
                                                          level_sets[domain_idx]);

      solution.update_ghost_values();

      for (const auto &cell_0 : system.locally_active_cell_iterators())
        if (cell_0->is_locally_owned() &&
            (mesh_classifiers[domain_idx]->location_to_level_set(cell_0->dealii_iterator()) ==
            NonMatching::LocationToLevelSet::intersected))
          {
            if (mesh_classifiers[domain_idx + 1]->location_to_level_set(cell_0->dealii_iterator()) == NonMatching::LocationToLevelSet::outside)
              continue;
            
            non_matching_fe_values.reinit(cell_0->dealii_iterator(),
                                          numbers::invalid_unsigned_int,
                                          numbers::invalid_unsigned_int,
                                          cell_0->active_fe_index());

            const double cell_side_length =
              cell_0->dealii_iterator()->minimum_vertex_distance();

            const unsigned int n_dofs_per_cell = fe[0].dofs_per_cell;

            std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);
            cell_0->get_dof_indices(dof_indices);

            Vector<Number> cell_vector_0(n_dofs_per_cell);
            Vector<Number> cell_vector_1(n_dofs_per_cell);

            // (II) surface integral to apply BC
            if (const auto &surface_fe_values_ptr =
                  non_matching_fe_values.get_surface_fe_values())
              {
                const auto &surface_fe_values = *surface_fe_values_ptr;

                std::vector<Number> quadrature_values_0(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_values(solution.block(domain_idx),
                                                      dof_indices,
                                                      quadrature_values_0);

                std::vector<Number> quadrature_values_1(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_values(solution.block(domain_idx + 1),
                                                      dof_indices,
                                                      quadrature_values_1);

                std::vector<Tensor<1, dim, Number>> quadrature_gradients_0(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_gradients(solution.block(domain_idx),
                                                        dof_indices,
                                                        quadrature_gradients_0);

                std::vector<Tensor<1, dim, Number>> quadrature_gradients_1(
                  surface_fe_values.n_quadrature_points);
                surface_fe_values.get_function_gradients(solution.block(domain_idx + 1),
                                                        dof_indices,
                                                        quadrature_gradients_1);


                for (const unsigned int q :
                    surface_fe_values.quadrature_point_indices())
                  {
                    const Point<dim> point= surface_fe_values.quadrature_point(q);
                        double c_surface= speed[domain_idx]->value(point);
                        double c_surface_other= speed[domain_idx + 1]->value(point);
                        double k_1 = c_surface_other/(c_surface + c_surface_other);
                        double k_2 = c_surface/(c_surface + c_surface_other);
                    const Tensor<1, dim> normal =
                      surface_fe_values.normal_vector(q);

                    // const auto tau_parameter = 0.5 * nitsche_parameter;
                    const auto tau_parameter = ((c_surface * c_surface_other)/(c_surface + c_surface_other)) * nitsche_parameter;

                    for (const unsigned int i : surface_fe_values.dof_indices())
                      {
                        const auto quadrature_value_jump =
                          (quadrature_values_0[q] - quadrature_values_1[q]);
                        // const auto quadrature_gradient_avg =
                        //   0.5 *
                        //   (quadrature_gradients_0[q] + quadrature_gradients_1[q]);
                        const auto quadrature_gradient_avg =
                          (k_1 * c_surface * quadrature_gradients_0[q] + k_2 * c_surface_other * quadrature_gradients_1[q]);

                        cell_vector_0(i) -=
                          (-k_1 * c_surface * normal * surface_fe_values.shape_grad(i, q) *
                            quadrature_value_jump -
                          surface_fe_values.shape_value(i, q) * normal *
                            quadrature_gradient_avg +
                          tau_parameter / cell_side_length *
                            surface_fe_values.shape_value(i, q) *
                            quadrature_value_jump) *
                          surface_fe_values.JxW(q);

                        cell_vector_1(i) -=
                          (-k_2 * c_surface_other * normal * surface_fe_values.shape_grad(i, q) *
                            quadrature_value_jump +
                          surface_fe_values.shape_value(i, q) * normal *
                            quadrature_gradient_avg -
                          tau_parameter / cell_side_length *
                            surface_fe_values.shape_value(i, q) *
                            quadrature_value_jump) *
                          surface_fe_values.JxW(q);
                      }
                  }
              }

            cell_0->get_dof_indices(dof_indices);
            constraints.distribute_local_to_global(cell_vector_0,
                                                  dof_indices,
                                                  vec_rhs.block(domain_idx));
            constraints.distribute_local_to_global(cell_vector_1,
                                                  dof_indices,
                                                  vec_rhs.block(domain_idx + 1));
          }
    }

    vec_rhs.compress(VectorOperation::add);
  }

private:
  const Discretization<dim, Number> &discretization;

  double ghost_parameter_A;
  double nitsche_parameter;

  std::shared_ptr<Function<dim>> function_domain_dbc;
  std::shared_ptr<Function<dim>> function_interface_dbc;
  std::shared_ptr<Function<dim>> function_rhs;
  std::vector<std::shared_ptr<Function<dim>>> speed;

  mutable std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> block_sparse_matrix;

  void
  compute_sparse_matrix(size_t & domain_idx, TrilinosWrappers::SparseMatrix &mat) const
  {
    std::shared_ptr<Function<dim>> required_speed = speed[domain_idx];
    // 0) extract information from discretization class
    const hp::MappingCollection<dim> &mapping = discretization.get_mapping();
    const Quadrature<1> &quadrature_1D = discretization.get_quadrature_1D();
    const Quadrature<dim - 1> &face_quadrature =
      discretization.get_face_quadrature();
    const GDM::System<dim>          &system = discretization.get_system();
    const AffineConstraints<Number> &constraints =
      discretization.get_affine_constraints();
    const std::vector<std::shared_ptr<NonMatching::MeshClassifier<dim>>> &mesh_classifiers =
      discretization.get_mesh_classifiers();
    const hp::FECollection<dim> &fe        = discretization.get_fe();
    const std::vector<VectorType>              &level_sets = discretization.get_level_sets();
    const DoFHandler<dim>       &level_set_dof_handler =
      discretization.get_level_set_dof_handler();

    AssertThrow(ghost_parameter_A != -1.0, ExcNotImplemented());
    AssertThrow(nitsche_parameter != -1.0, ExcNotImplemented());

    // 1) create sparsity pattern
    if (mat.m() == 0 || mat.n() == 0)
      {
        TrilinosWrappers::SparsityPattern sparsity_pattern;
        sparsity_pattern.reinit(system.locally_owned_dofs(), MPI_COMM_WORLD);
        system.create_flux_sparsity_pattern(constraints, sparsity_pattern);
        sparsity_pattern.compress();

        mat.reinit(sparsity_pattern);
      }
    else
      {
        mat = 0.0;
      }

    const auto face_has_ghost_penalty = [&](const auto        &cell,
                                            const unsigned int face_index) {
      if (cell->at_boundary(face_index))
        return false;

      const NonMatching::LocationToLevelSet cell_location =
        mesh_classifiers[domain_idx]->location_to_level_set(cell);

      const NonMatching::LocationToLevelSet neighbor_location =
        mesh_classifiers[domain_idx]->location_to_level_set(cell->neighbor(face_index));

      if (cell_location == NonMatching::LocationToLevelSet::intersected &&
          neighbor_location != NonMatching::LocationToLevelSet::outside)
        return true;

      if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
          cell_location != NonMatching::LocationToLevelSet::outside)
        return true;

      return false;
    };

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      *mesh_classifiers[domain_idx],
                                                      level_set_dof_handler,
                                                      level_sets[domain_idx]);

    NonMatching::RegionUpdateFlags region_update_flags_face;
    region_update_flags_face.inside =
        update_values | update_gradients | update_JxW_values |
        update_quadrature_points | update_normal_vectors;

    NonMatching::FEInterfaceValues<dim> non_matching_fe_interface_values(
      fe,
      quadrature_1D,
      region_update_flags_face,
      *mesh_classifiers[domain_idx],
      level_set_dof_handler,
      level_sets[domain_idx]);
                                                        
    FEInterfaceValues<dim> fe_interface_values(
      mapping,
      fe,
      hp::QCollection<dim - 1>(face_quadrature),
      update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points);

    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : system.locally_active_cell_iterators())
      if (cell->is_locally_owned() &&
          (mesh_classifiers[domain_idx]->location_to_level_set(cell->dealii_iterator()) !=
           NonMatching::LocationToLevelSet::outside))
        {
          non_matching_fe_values.reinit(cell->dealii_iterator(),
                                        numbers::invalid_unsigned_int,
                                        numbers::invalid_unsigned_int,
                                        cell->active_fe_index());

          const double cell_side_length =
            cell->dealii_iterator()->minimum_vertex_distance();

          const unsigned int dofs_per_cell = fe[0].dofs_per_cell;

          // compute element stiffness matrix
          FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);

          // (I) cell integral
          if (const auto &fe_values =
                non_matching_fe_values.get_inside_fe_values())
            {
              for (const unsigned int q_index :
                   fe_values->quadrature_point_indices())
                {
                  const Point<dim> point= fe_values->quadrature_point(q_index);
                    double speed_cell= required_speed->value(point);
                  for (const unsigned int i : fe_values->dof_indices())
                    for (const unsigned int j : fe_values->dof_indices())
                      // (∇v, ∇u)
                      cell_matrix(i, j) += speed_cell * fe_values->shape_grad(i, q_index) *
                                           fe_values->shape_grad(j, q_index) *
                                           fe_values->JxW(q_index);
                }
            }

          // (II) surface integral to apply BC
          if (function_interface_dbc)
            if (const auto &surface_fe_values_ptr =
                  non_matching_fe_values.get_surface_fe_values())
              {
                const auto &surface_fe_values = *surface_fe_values_ptr;
                for (const unsigned int q :
                     surface_fe_values.quadrature_point_indices())
                  {
                    const Point<dim> point= surface_fe_values.quadrature_point(q);
                        double c_surface= required_speed->value(point);
                    const Tensor<1, dim> &normal =
                      surface_fe_values.normal_vector(q);
                    for (const unsigned int i : surface_fe_values.dof_indices())
                      for (const unsigned int j :
                           surface_fe_values.dof_indices())
                        {
                          // - <v, ∂u/∂n> - <∂v/∂n, u> + γ_D/h <v, u>
                          cell_matrix(i, j) +=
                            (-normal * surface_fe_values.shape_grad(i, q) *
                               surface_fe_values.shape_value(j, q) +
                             -normal * surface_fe_values.shape_grad(j, q) *
                               surface_fe_values.shape_value(i, q) +
                             nitsche_parameter / cell_side_length *
                               surface_fe_values.shape_value(i, q) *
                               surface_fe_values.shape_value(j, q)) * c_surface *
                            surface_fe_values.JxW(q);
                        }
                  }
              }

          // (III) face integral for apply DBC
          if (function_domain_dbc)
            for (const auto f : cell->dealii_iterator()->face_indices())
              if (cell->dealii_iterator()->face(f)->at_boundary())
                {
                  non_matching_fe_interface_values.reinit(
                    cell->dealii_iterator(),
                    f,
                    numbers::invalid_unsigned_int,
                    numbers::invalid_unsigned_int,
                    cell->active_fe_index());

                  if (const auto &surface_fe_values_ptr = non_matching_fe_interface_values.get_inside_fe_values())
                    {
                      const auto &surface_fe_values =
                        surface_fe_values_ptr->get_fe_face_values(0);
                      for (const unsigned int q :
                           surface_fe_values.quadrature_point_indices())
                        {
                          const Point<dim> &point =
                            surface_fe_values.quadrature_point(q);
                          const Tensor<1, dim> normal =
                            surface_fe_values.normal_vector(q);
                          double c_surface = required_speed->value(point);

                          for (const unsigned int i : surface_fe_values.dof_indices())
                          {
                            for (const unsigned int j : surface_fe_values.dof_indices())
                              {
                                  cell_matrix(i, j) +=
                                    (-normal * surface_fe_values.shape_grad(i, q) *
                                      surface_fe_values.shape_value(j, q) +
                                    -normal * surface_fe_values.shape_grad(j, q) *
                                      surface_fe_values.shape_value(i, q) +
                                    nitsche_parameter / cell_side_length *
                                      surface_fe_values.shape_value(i, q) *
                                      surface_fe_values.shape_value(j, q)) * c_surface *
                                    surface_fe_values.JxW(q);
                              }
                          }
                        }
                    }
                }

          // (II) face integral to apply GP
          for (const unsigned int f : cell->dealii_iterator()->face_indices())
            if (face_has_ghost_penalty(cell->dealii_iterator(), f))
              {
                fe_interface_values.reinit(
                  cell->dealii_iterator(),
                  f,
                  numbers::invalid_unsigned_int,
                  cell->dealii_iterator()->neighbor(f),
                  cell->dealii_iterator()->neighbor_of_neighbor(f),
                  numbers::invalid_unsigned_int,
                  numbers::invalid_unsigned_int,
                  numbers::invalid_unsigned_int,
                  cell->active_fe_index(),
                  cell->neighbor(f)->active_fe_index());

                const unsigned int n_interface_dofs =
                  fe_interface_values.n_current_interface_dofs();
                FullMatrix<double> local_stabilization(n_interface_dofs,
                                                       n_interface_dofs);
                for (unsigned int q = 0;
                     q < fe_interface_values.n_quadrature_points;
                     ++q)
                  {
                    const Tensor<1, dim> normal = fe_interface_values.normal(q);
                    const Point<dim> point= fe_interface_values.quadrature_point(q);
                    double c_interface = required_speed->value(point);
                    for (unsigned int i = 0; i < n_interface_dofs; ++i)
                      for (unsigned int j = 0; j < n_interface_dofs; ++j)
                        {
                          // clang-format off
                          // γ_A j(v, u) / h^2 with j(v, u)= ∑ h^3 <∂v/∂n, ∂u/∂n>
                          local_stabilization(i, j) +=
                            .5 * ghost_parameter_A * c_interface * cell_side_length *
                            cell_side_length * cell_side_length *
                            (normal * fe_interface_values.jump_in_shape_gradients(i, q)) *
                            (normal * fe_interface_values.jump_in_shape_gradients(j, q)) *
                            fe_interface_values.JxW(q);
                          // clang-format on
                        }
                  }

                std::vector<types::global_dof_index>
                  local_interface_dof_indices;
                dof_indices.resize(dofs_per_cell);
                cell->get_dof_indices(dof_indices);
                for (const auto i : dof_indices)
                  local_interface_dof_indices.emplace_back(i);
                cell->neighbor(f)->get_dof_indices(dof_indices);
                for (const auto i : dof_indices)
                  local_interface_dof_indices.emplace_back(i);

                mat.add(local_interface_dof_indices,
                                  local_stabilization);
              }

          // get indices
          dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(dof_indices);

          // assemble
          constraints.distribute_local_to_global(cell_matrix,
                                                 dof_indices,
                                                 mat);
        }

    mat.compress(VectorOperation::values::add);

    for (auto &entry : mat)
      if ((entry.row() == entry.column()) && (entry.value() == 0.0))
        entry.value() = 1.0;
  }
};
