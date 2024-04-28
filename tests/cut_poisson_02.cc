// Test cut Poisson problem: serial, FEM.

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <gdm/data_out.h>
#include <gdm/system.h>

#include <fstream>
#include <vector>


using namespace dealii;

template <int dim>
class AnalyticalSolution : public Function<dim>
{
public:
  double
  value(const Point<dim>  &point,
        const unsigned int component = 0) const override
  {
    AssertIndexRange(component, this->n_components);
    (void)component;

    return 1. - 2. / dim * (point.norm_square() - 1.);
  }
};



template <int dim>
void
test()
{
  const unsigned int n_subdivisions      = 64;
  const unsigned int n_components        = 1;
  const unsigned int fe_degree           = 3;
  const unsigned int fe_degree_level_set = 1;
  const unsigned int fe_degree_output    = 2;

  ConvergenceTable convergence_table;

  const Functions::ConstantFunction<dim> rhs_function(4.0);
  const Functions::ConstantFunction<dim> boundary_condition(1.0);

  // Create GDM system
  GDM::System<dim> system(fe_degree, n_components);

  // Create mesh
  system.subdivided_hyper_cube(n_subdivisions, -1.21, 1.21);

  // Create finite elements
  const auto &fe = system.get_fe();

  // Create mapping
  hp::MappingCollection<dim> mapping;
  mapping.push_back(MappingQ1<dim>());

  // Create constraints
  const AffineConstraints<double> constraints;

  // Categorize cells
  system.categorize();

  const auto &triangulation = system.get_triangulation();


  // level set and classify cells
  const FE_Q<dim> fe_level_set(fe_degree_level_set);
  DoFHandler<dim> level_set_dof_handler(triangulation);
  level_set_dof_handler.distribute_dofs(fe_level_set);

  Vector<double> level_set;
  level_set.reinit(level_set_dof_handler.n_dofs());

  NonMatching::MeshClassifier<dim> mesh_classifier(level_set_dof_handler,
                                                   level_set);

  const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
  VectorTools::interpolate(level_set_dof_handler,
                           signed_distance_sphere,
                           level_set);

  mesh_classifier.reclassify();

  // allocate memory
  DynamicSparsityPattern dsp(system.n_dofs());
  system.create_sparsity_pattern(constraints, dsp);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> stiffness_matrix;
  stiffness_matrix.reinit(sparsity_pattern);

  Vector<double> solution, rhs;
  solution.reinit(system.n_dofs());
  rhs.reinit(system.n_dofs());

  // assemble system
  const unsigned int n_dofs_per_cell = fe[0].dofs_per_cell;
  FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
  Vector<double>     local_rhs(n_dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

  const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;

  const QGauss<1> quadrature_1D(fe_degree + 1);

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside = update_values | update_gradients |
                               update_JxW_values | update_quadrature_points;
  region_update_flags.surface = update_values | update_gradients |
                                update_JxW_values | update_quadrature_points |
                                update_normal_vectors;

  NonMatching::FEValues<dim> non_matching_fe_values(fe,
                                                    quadrature_1D,
                                                    region_update_flags,
                                                    mesh_classifier,
                                                    level_set_dof_handler,
                                                    level_set);

  for (const auto &cell : system.locally_active_cell_iterators())
    if (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
        NonMatching::LocationToLevelSet::outside)
      {
        local_stiffness = 0;
        local_rhs       = 0;

        const double cell_side_length =
          cell->dealii_iterator()->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell->dealii_iterator(),
                                      numbers::invalid_unsigned_int,
                                      numbers::invalid_unsigned_int,
                                      cell->active_fe_index());

        const std::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (inside_fe_values)
          for (const unsigned int q :
               inside_fe_values->quadrature_point_indices())
            {
              const Point<dim> &point = inside_fe_values->quadrature_point(q);
              for (const unsigned int i : inside_fe_values->dof_indices())
                {
                  for (const unsigned int j : inside_fe_values->dof_indices())
                    {
                      local_stiffness(i, j) +=
                        inside_fe_values->shape_grad(i, q) *
                        inside_fe_values->shape_grad(j, q) *
                        inside_fe_values->JxW(q);
                    }
                  local_rhs(i) += rhs_function.value(point) *
                                  inside_fe_values->shape_value(i, q) *
                                  inside_fe_values->JxW(q);
                }
            }

        const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
          {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);
                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    for (const unsigned int j :
                         surface_fe_values->dof_indices())
                      {
                        local_stiffness(i, j) +=
                          (-normal * surface_fe_values->shape_grad(i, q) *
                             surface_fe_values->shape_value(j, q) +
                           -normal * surface_fe_values->shape_grad(j, q) *
                             surface_fe_values->shape_value(i, q) +
                           nitsche_parameter / cell_side_length *
                             surface_fe_values->shape_value(i, q) *
                             surface_fe_values->shape_value(j, q)) *
                          surface_fe_values->JxW(q);
                      }
                    local_rhs(i) +=
                      boundary_condition.value(point) *
                      (nitsche_parameter / cell_side_length *
                         surface_fe_values->shape_value(i, q) -
                       normal * surface_fe_values->shape_grad(i, q)) *
                      surface_fe_values->JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);

        stiffness_matrix.add(local_dof_indices, local_stiffness);
        rhs.add(local_dof_indices, local_rhs);
      }

  for (auto &entry : stiffness_matrix)
    if ((entry.row() == entry.column()) && (entry.value() == 0.0))
      {
        entry.value() = 1.0;
      }

  // solve system
  const unsigned int max_iterations = solution.size();
  ReductionControl   solver_control(max_iterations, 1.e-10, 1.e-6);
  SolverCG<>         solver(solver_control);
  solver.solve(stiffness_matrix, solution, rhs, PreconditionIdentity());


  // create Paraview output
  GDM::DataOut<dim> data_out(system, mapping, fe_degree_output);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream file("step-85.vtu");
  data_out.write_vtu(file);

  // compute error
  const QGauss<1> quadrature_1D_error(fe_degree + 1);

  NonMatching::RegionUpdateFlags region_update_flags_error;
  region_update_flags_error.inside =
    update_values | update_JxW_values | update_quadrature_points;

  NonMatching::FEValues<dim> non_matching_fe_values_error(
    fe,
    quadrature_1D_error,
    region_update_flags_error,
    mesh_classifier,
    level_set_dof_handler,
    level_set);

  AnalyticalSolution<dim> analytical_solution;
  double                  error_L2_squared = 0;

  for (const auto &cell : system.locally_active_cell_iterators())
    if (mesh_classifier.location_to_level_set(cell->dealii_iterator()) !=
        NonMatching::LocationToLevelSet::outside)
      {
        non_matching_fe_values_error.reinit(cell->dealii_iterator(),
                                            numbers::invalid_unsigned_int,
                                            numbers::invalid_unsigned_int,
                                            cell->active_fe_index());

        const std::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values_error.get_inside_fe_values();

        if (fe_values)
          {
            std::vector<double> solution_values(fe_values->n_quadrature_points);

            cell->get_dof_indices(local_dof_indices);
            fe_values->get_function_values(solution,
                                           local_dof_indices,
                                           solution_values);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values.at(q) - analytical_solution.value(point);
                error_L2_squared +=
                  Utilities::fixed_power<2>(error_at_point) * fe_values->JxW(q);
              }
          }
      }

  const double error_L2 = std::sqrt(error_L2_squared);
  const double cell_side_length =
    triangulation.begin_active()->minimum_vertex_distance();

  convergence_table.add_value("Mesh size", cell_side_length);
  convergence_table.add_value("L2-Error", error_L2);

  convergence_table.set_scientific("L2-Error", true);

  std::cout << std::endl;
  convergence_table.write_text(std::cout);
  std::cout << std::endl;
}



int
main()
{
  test<2>();
}
