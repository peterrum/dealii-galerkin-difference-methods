#include <deal.II/base/convergence_table.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/time_stepping.templates.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <gdm/fe.h>


using namespace dealii;

TimeStepping::runge_kutta_method
get_runge_kutta_method(const unsigned int n_stages)
{
  if (n_stages == 1)
    return TimeStepping::runge_kutta_method::FORWARD_EULER;
  else if (n_stages == 2)
    return TimeStepping::runge_kutta_method::HEUN_EULER;
  else if (n_stages == 3)
    return TimeStepping::runge_kutta_method::RK_THIRD_ORDER;
  else if (n_stages == 4)
    return TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;
  else if (n_stages == 5)
    return TimeStepping::runge_kutta_method::RK_FIFTH_ORDER;
  else if (n_stages == 6)
    return TimeStepping::runge_kutta_method::RK_SIXTH_ORDER;

  AssertThrow(false, ExcNotImplemented());

  return {};
}



namespace dealii::TimeStepping
{
  /**
   * MyExplicitRungeKutta is derived from RungeKutta and implement the explicit
   * methods.
   *
   * This is a modified version of dealii::TimeStepping::ExplicitRungeKutta,
   * since it does not support (version 9.7) Heun's method.
   *
   * See: https://github.com/dealii/dealii/pull/19639
   */
  template <typename VectorType>
  class MyExplicitRungeKutta : public RungeKutta<VectorType>
  {
  public:
    using RungeKutta<VectorType>::evolve_one_time_step;

    /**
     * Default constructor. This constructor creates an object for which
     * you will want to call <code>initialize(runge_kutta_method)</code>
     * before it can be used.
     */
    MyExplicitRungeKutta() = default;

    /**
     * Constructor. This function calls initialize(runge_kutta_method).
     */
    MyExplicitRungeKutta(const runge_kutta_method method);

    /**
     * Initialize the explicit Runge-Kutta method.
     */
    void
    initialize(const runge_kutta_method method) override;

    /**
     * This function is used to advance from time @p t to t+ @p delta_t. @p f
     * is the function $ f(t,y) $ that should be integrated, the input
     * parameters are the time t and the vector y and the output is value of f
     * at this point. @p id_minus_tau_J_inverse is a function that computes $
     * inv(I-\tau J)$ where $ I $ is the identity matrix, $ \tau $ is given,
     * and $ J $ is the Jacobian $ \frac{\partial f}{\partial y} $. The input
     * parameter are the time, $ \tau $, and a vector. The output is the value
     * of function at this point. evolve_one_time_step returns the time at the
     * end of the time step.
     *
     * @note @p id_minus_tau_J_inverse is ignored since the method is explicit.
     */
    double
    evolve_one_time_step(
      const std::function<VectorType(const double, const VectorType &)> &f,
      const std::function<
        VectorType(const double, const double, const VectorType &)>
                 &id_minus_tau_J_inverse,
      double      t,
      double      delta_t,
      VectorType &y) override;

    /**
     * This function is used to advance from time @p t to t+ @p delta_t. This
     * function is similar to the one derived from RungeKutta, but does not
     * required id_minus_tau_J_inverse because it is not used for explicit
     * methods. evolve_one_time_step returns the time at the end of the time
     * step.
     */
    double
    evolve_one_time_step(
      const std::function<VectorType(const double, const VectorType &)> &f,
      double                                                             t,
      double      delta_t,
      VectorType &y);

    /**
     * This structure stores the name of the method used.
     */
    struct Status : public TimeStepping<VectorType>::Status
    {
      Status()
        : method(invalid)
      {}

      runge_kutta_method method;
    };

    /**
     * Return the status of the current object.
     */
    const Status &
    get_status() const override;

  private:
    /**
     * Compute the different stages needed.
     */
    void
    compute_stages(
      const std::function<VectorType(const double, const VectorType &)> &f,
      const double                                                       t,
      const double             delta_t,
      const VectorType        &y,
      std::vector<VectorType> &f_stages) const;

    /**
     * Status structure of the object.
     */
    Status status;
  };

} // namespace dealii::TimeStepping



namespace dealii::TimeStepping
{
  template <typename VectorType>
  MyExplicitRungeKutta<VectorType>::MyExplicitRungeKutta(
    const runge_kutta_method method)
  {
    // virtual functions called in constructors and destructors never use the
    // override in a derived class
    // for clarity be explicit on which function is called
    MyExplicitRungeKutta<VectorType>::initialize(method);
  }



  template <typename VectorType>
  void
  MyExplicitRungeKutta<VectorType>::initialize(const runge_kutta_method method)
  {
    status.method = method;

    switch (method)
      {
        case (FORWARD_EULER):
          {
            this->n_stages = 1;
            this->b.push_back(1.0);
            this->c.push_back(0.0);

            break;
          }
        case (HEUN_EULER):
          {
            this->n_stages = 2;
            this->b.push_back(1.0 / 2.0);
            this->b.push_back(1.0 / 2.0);
            this->c.push_back(0.0);
            this->c.push_back(1.0);

            std::vector<double> tmp;
            this->a.push_back(tmp);
            tmp.resize(1);
            tmp[0] = 1.0;
            this->a.push_back(tmp);

            break;
          }
        case (RK_THIRD_ORDER):
          {
            this->n_stages = 3;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            this->b.push_back(1.0 / 6.0);
            this->b.push_back(2.0 / 3.0);
            this->b.push_back(1.0 / 6.0);
            this->c.push_back(0.0);
            this->c.push_back(0.5);
            this->c.push_back(1.0);
            std::vector<double> tmp;
            this->a.push_back(tmp);
            tmp.resize(1);
            tmp[0] = 0.5;
            this->a.push_back(tmp);
            tmp.resize(2);
            tmp[0] = -1.0;
            tmp[1] = 2.0;
            this->a.push_back(tmp);

            break;
          }
        case (SSP_THIRD_ORDER):
          {
            this->n_stages = 3;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            this->b.push_back(1.0 / 6.0);
            this->b.push_back(1.0 / 6.0);
            this->b.push_back(2.0 / 3.0);
            this->c.push_back(0.0);
            this->c.push_back(1.0);
            this->c.push_back(0.5);
            std::vector<double> tmp;
            this->a.push_back(tmp);
            tmp.resize(1);
            tmp[0] = 1.0;
            this->a.push_back(tmp);
            tmp.resize(2);
            tmp[0] = 1.0 / 4.0;
            tmp[1] = 1.0 / 4.0;
            this->a.push_back(tmp);

            break;
          }
        case (RK_CLASSIC_FOURTH_ORDER):
          {
            this->n_stages = 4;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            std::vector<double> tmp;
            this->a.push_back(tmp);
            tmp.resize(1);
            tmp[0] = 0.5;
            this->a.push_back(tmp);
            tmp.resize(2);
            tmp[0] = 0.0;
            tmp[1] = 0.5;
            this->a.push_back(tmp);
            tmp.resize(3);
            tmp[1] = 0.0;
            tmp[2] = 1.0;
            this->a.push_back(tmp);
            this->b.push_back(1.0 / 6.0);
            this->b.push_back(1.0 / 3.0);
            this->b.push_back(1.0 / 3.0);
            this->b.push_back(1.0 / 6.0);
            this->c.push_back(0.0);
            this->c.push_back(0.5);
            this->c.push_back(0.5);
            this->c.push_back(1.0);

            break;
          }
        case (RK_FIFTH_ORDER):
          {
            /**
             * Rabiei, F. and Ismail, F., 2012. Fifth-order improved
             * Runge-Kutta method with reduced number of function evaluations.
             * Australian Journal of Basic and Applied Sciences, 6(3),
             * pp.97-105.
             *
             * Hossain, M.B., Hossain, M.J., Miah, M.M. and Alam, M.S., 2017.
             * A comparative study on fourth order and butcher’s fifth order
             * runge-kutta methods with third order initial value problem (IVP).
             * Appl. Comput. Math, 6(6), p.243.
             */
            this->n_stages = 6;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            std::vector<double> tmp;
            this->a.push_back(tmp);
            tmp.assign(5, 0.0);
            tmp[0] = 1.0 / 4.0;
            this->a.push_back(tmp);
            tmp.assign(5, 0.0);
            tmp[0] = 1.0 / 8.0;
            tmp[1] = 1.0 / 8.0;
            this->a.push_back(tmp);
            tmp.assign(5, 0.0);
            tmp[0] = 0.0;
            tmp[1] = -1.0 / 2.0;
            tmp[2] = 1.0;
            this->a.push_back(tmp);
            tmp.assign(5, 0.0);
            tmp[0] = 3.0 / 16.0;
            tmp[1] = 0.0;
            tmp[2] = 0.0;
            tmp[3] = 9.0 / 16.0;
            this->a.push_back(tmp);
            tmp.assign(5, 0.0);
            tmp[0] = -3.0 / 7.0;
            tmp[1] = 2.0 / 7.0;
            tmp[2] = 12.0 / 7.0;
            tmp[3] = -12.0 / 7.0;
            tmp[4] = 8.0 / 7.0;
            this->a.push_back(tmp);

            this->b.push_back(7.0 / 90.0);
            this->b.push_back(0.0);
            this->b.push_back(32.0 / 90.0);
            this->b.push_back(12.0 / 90.0);
            this->b.push_back(32.0 / 90.0);
            this->b.push_back(7.0 / 90.0);

            this->c.push_back(0.0);
            this->c.push_back(1.0 / 4.0);
            this->c.push_back(1.0 / 4.0);
            this->c.push_back(1.0 / 2.0);
            this->c.push_back(3.0 / 4.0);
            this->c.push_back(1.0);

            break;
          }
        case (RK_SIXTH_ORDER):
          {
            /**
             * Butcher, J.C., 1964. On Runge-Kutta processes of high order.
             * Journal of the Australian Mathematical Society, 4(2), pp.179-194.
             */
            this->n_stages = 7;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            std::vector<double> tmp;
            this->a.push_back(tmp);
            tmp.assign(6, 0.0);
            tmp[0] = 1.0 / 3.0;
            this->a.push_back(tmp);
            tmp.assign(6, 0.0);
            tmp[0] = 0.0;
            tmp[1] = 2.0 / 3.0;
            this->a.push_back(tmp);
            tmp.assign(6, 0.0);
            tmp[0] = 1.0 / 12.0;
            tmp[1] = 1.0 / 3.0;
            tmp[2] = -1.0 / 12.0;
            this->a.push_back(tmp);
            tmp.assign(6, 0.0);
            tmp[0] = -1.0 / 16.0;
            tmp[1] = 9.0 / 8.0;
            tmp[2] = -3.0 / 16.0;
            tmp[3] = -3.0 / 8.0;
            this->a.push_back(tmp);
            tmp.assign(6, 0.0);
            tmp[0] = 0.0;
            tmp[1] = 9.0 / 8.0;
            tmp[2] = -3.0 / 8.0;
            tmp[3] = -3.0 / 4.0;
            tmp[4] = 1.0 / 2.0;
            this->a.push_back(tmp);
            tmp.assign(6, 0.0);
            tmp[0] = 9.0 / 44.0;
            tmp[1] = -9.0 / 11.0;
            tmp[2] = 63.0 / 44.0;
            tmp[3] = 18.0 / 11.0;
            tmp[4] = 0.0;
            tmp[5] = -16.0 / 11.0;
            this->a.push_back(tmp);

            this->b.push_back(11.0 / 120.0);
            this->b.push_back(0.0);
            this->b.push_back(27.0 / 40.0);
            this->b.push_back(27.0 / 40.0);
            this->b.push_back(-4.0 / 15.0);
            this->b.push_back(-4.0 / 15.0);
            this->b.push_back(11.0 / 120.0);

            this->c.push_back(0.0);
            this->c.push_back(1.0 / 3.0);
            this->c.push_back(2.0 / 3.0);
            this->c.push_back(1.0 / 3.0);
            this->c.push_back(1.0 / 2.0);
            this->c.push_back(1.0 / 2.0);
            this->c.push_back(1.0);

            break;
          }
        default:
          {
            AssertThrow(
              false, ExcMessage("Unimplemented explicit Runge-Kutta method."));
          }
      }
  }



  template <typename VectorType>
  double
  MyExplicitRungeKutta<VectorType>::evolve_one_time_step(
    const std::function<VectorType(const double, const VectorType &)> &f,
    const std::function<
      VectorType(const double, const double, const VectorType &)>
      & /*id_minus_tau_J_inverse*/,
    double      t,
    double      delta_t,
    VectorType &y)
  {
    return evolve_one_time_step(f, t, delta_t, y);
  }



  template <typename VectorType>
  double
  MyExplicitRungeKutta<VectorType>::evolve_one_time_step(
    const std::function<VectorType(const double, const VectorType &)> &f,
    double                                                             t,
    double                                                             delta_t,
    VectorType                                                        &y)
  {
    Assert(status.method != runge_kutta_method::invalid, ExcNoMethodSelected());

    std::vector<VectorType> f_stages(this->n_stages, y);
    // Compute the different stages needed.
    compute_stages(f, t, delta_t, y, f_stages);

    // Linear combinations of the stages.
    for (unsigned int i = 0; i < this->n_stages; ++i)
      y.sadd(1., delta_t * this->b[i], f_stages[i]);

    return (t + delta_t);
  }



  template <typename VectorType>
  const typename MyExplicitRungeKutta<VectorType>::Status &
  MyExplicitRungeKutta<VectorType>::get_status() const
  {
    return status;
  }



  template <typename VectorType>
  void
  MyExplicitRungeKutta<VectorType>::compute_stages(
    const std::function<VectorType(const double, const VectorType &)> &f,
    const double                                                       t,
    const double                                                       delta_t,
    const VectorType                                                  &y,
    std::vector<VectorType> &f_stages) const
  {
    for (unsigned int i = 0; i < this->n_stages; ++i)
      {
        VectorType Y(y);
        for (unsigned int j = 0; j < i; ++j)
          Y.sadd(1., delta_t * this->a[i][j], f_stages[j]);
        // Evaluate the function f at the point (t+c[i]*delta_t,Y).
        f_stages[i] = f(t + this->c[i] * delta_t, Y);
      }
  }

} // namespace dealii::TimeStepping


template <int dim>
void
test(const unsigned int n_subdivisions_1D,
     const unsigned int fe_degree,
     ConvergenceTable  &table)
{
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  const unsigned int n_ghost_cells = fe_degree / 2;
  const double       dx            = 2.0 / n_subdivisions_1D;
  const double       cfl           = 0.2;
  const double       delta_t       = dx * cfl;

  const TimeStepping::runge_kutta_method runge_kutta_method =
    get_runge_kutta_method(fe_degree + 1);

  MappingQ1<dim> mapping;
  QGauss<dim>    quadrature(fe_degree + 1);

  AffineConstraints<double> constraints;
  constraints.close();

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria,
                                       n_subdivisions_1D + 2 * n_ghost_cells,
                                       -1.0 - dx * n_ghost_cells,
                                       +1.0 + dx * n_ghost_cells);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(1));

  const auto hp_fe =
    GDM::generate_fe_collection<dim>(GDM::generate_polynomials_1D(fe_degree),
                                     1);

  const auto        &fe              = hp_fe[n_ghost_cells];
  const unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();

  const auto exact_solution =
    std::make_shared<ScalarFunctionFromFunctionObject<dim>>(
      [](const auto t, const auto &p) {
        const auto r = p.norm();

        if (dim == 1)
          {
            const auto wave_number = 1.0 * numbers::PI;
            return std::cos(wave_number * r) * std::cos(wave_number * t);
          }
        else
          AssertThrow(false, ExcNotImplemented());
      });

  BlockVectorType solution(2);
  solution.block(0).reinit(dof_handler.n_dofs());
  solution.block(1).reinit(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, *exact_solution, solution.block(0));

  TrilinosWrappers::SparsityPattern sparsity_pattern;
  TrilinosWrappers::SparseMatrix    mass_matrix;
  TrilinosWrappers::SparseMatrix    stiffness_matrix;

  // allocate memory for mass matrix
  {
    sparsity_pattern.reinit(dof_handler.n_dofs(), dof_handler.n_dofs());

    std::vector<types::global_dof_index> dof_indices;
    for (const auto &cell : tria.active_cell_iterators())
      if ((cell->active_cell_index() >= n_ghost_cells) &&
          (cell->active_cell_index() < n_subdivisions_1D + n_ghost_cells))
        {
          dof_indices.resize(n_dofs_per_cell);

          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            dof_indices[i] = cell->active_cell_index() - n_ghost_cells + i;

          constraints.add_entries_local_to_global(dof_indices,
                                                  sparsity_pattern);
        }

    sparsity_pattern.compress();

    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
  }

  // compute mass and stiffness matrix
  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_JxW_values | update_values | update_gradients);


  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : tria.active_cell_iterators())
    if ((cell->active_cell_index() >= n_ghost_cells) &&
        (cell->active_cell_index() < n_subdivisions_1D + n_ghost_cells))
      {
        fe_values.reinit(cell);

        FullMatrix<double> mass_cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
        FullMatrix<double> stiffness_cell_matrix(n_dofs_per_cell,
                                                 n_dofs_per_cell);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              mass_cell_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                        fe_values.shape_value(j, q_index) *
                                        fe_values.JxW(q_index);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              stiffness_cell_matrix(i, j) -= fe_values.shape_grad(i, q_index) *
                                             fe_values.shape_grad(j, q_index) *
                                             fe_values.JxW(q_index);

        // get indices
        dof_indices.resize(n_dofs_per_cell);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          dof_indices[i] = cell->active_cell_index() - n_ghost_cells + i;

        constraints.distribute_local_to_global(mass_cell_matrix,
                                               dof_indices,
                                               mass_matrix);
        constraints.distribute_local_to_global(stiffness_cell_matrix,
                                               dof_indices,
                                               stiffness_matrix);
      }

  mass_matrix.compress(VectorOperation::values::add);
  stiffness_matrix.compress(VectorOperation::values::add);

  // create direct solver
  TrilinosWrappers::SolverDirect mass_matrix_solver;
  mass_matrix_solver.initialize(mass_matrix);



  // Define rhs of ODE
  const auto fu_rhs = [&](const double time, const BlockVectorType &solution) {
    (void)time;

    BlockVectorType result;
    result.reinit(solution);
    VectorType vec_rhs;
    vec_rhs.reinit(solution.block(0));

    // du/dt = v
    result.block(0) = solution.block(1);

    // dv/dt = f(t, u)
    stiffness_matrix.vmult(vec_rhs, solution.block(0));
    mass_matrix_solver.solve(result.block(1), vec_rhs);

    return result;
  };

  unsigned int counter = 0;
  double       error   = 0.0;

  const auto postprocess = [&](const double time, const VectorType &solution) {
    if (true)
      {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;

        DataOut<dim> data_out;
        data_out.set_flags(flags);
        data_out.attach_triangulation(tria);
        data_out.add_data_vector(dof_handler, solution, "solution_lin");

        data_out.set_cell_selection(
          [&](const typename Triangulation<dim>::cell_iterator &cell) {
            return (cell->active_cell_index() >= n_ghost_cells) &&
                   (cell->active_cell_index() <
                    n_subdivisions_1D + n_ghost_cells);
          });

        DoFHandler<dim> dof_handler_solution;
        if (exact_solution)
          {
            exact_solution->set_time(time);

            dof_handler_solution.reinit(tria);
            dof_handler_solution.distribute_dofs(FE_Q<dim>(fe_degree));

            LinearAlgebra::distributed::Vector<double> analytical_solution(
              dof_handler_solution.n_dofs());
            VectorTools::interpolate(mapping,
                                     dof_handler_solution,
                                     *exact_solution,
                                     analytical_solution);
            data_out.add_data_vector(dof_handler_solution,
                                     analytical_solution,
                                     "solution_ana");

            analytical_solution = 0.0;

            FEValues<dim> fe_values(
              mapping,
              fe,
              Quadrature<dim>(
                dof_handler_solution.get_fe().get_unit_support_points()),
              update_values);

            for (const auto &cell : tria.active_cell_iterators())
              if ((cell->active_cell_index() >= n_ghost_cells) &&
                  (cell->active_cell_index() <
                   n_subdivisions_1D + n_ghost_cells))
                {
                  fe_values.reinit(cell);

                  std::vector<types::global_dof_index> dof_indices(
                    fe_values.dofs_per_cell);

                  for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
                    dof_indices[i] =
                      cell->active_cell_index() - n_ghost_cells + i;

                  std::vector<double> quadrature_values(
                    fe_values.n_quadrature_points);
                  fe_values.get_function_values(solution,
                                                dof_indices,
                                                quadrature_values);

                  cell->as_dof_handler_iterator(dof_handler_solution)
                    ->get_dof_indices(dof_indices);

                  for (unsigned int i = 0; i < dof_indices.size(); ++i)
                    analytical_solution[dof_indices[i]] = quadrature_values[i];
                }

            data_out.add_data_vector(dof_handler_solution,
                                     analytical_solution,
                                     "solution");
          }

        const std::string file_name =
          "results_" + std::to_string(counter) + ".vtu";
        counter++;

        // write data
        data_out.build_patches(
          mapping,
          fe_degree,
          DataOut<dim>::CurvedCellRegion::curved_inner_cells);
        data_out.write_vtu_in_parallel(file_name,
                                       dof_handler.get_mpi_communicator());
      }


    if (exact_solution)
      {
        exact_solution->set_time(time);

        FEValues<dim> fe_values(mapping,
                                fe,
                                QGauss<dim>(fe_degree + 3),
                                update_values | update_JxW_values |
                                  update_quadrature_points);

        error = 0.0;

        for (const auto &cell : tria.active_cell_iterators())
          if ((cell->active_cell_index() >= n_ghost_cells) &&
              (cell->active_cell_index() < n_subdivisions_1D + n_ghost_cells))
            {
              fe_values.reinit(cell);

              std::vector<types::global_dof_index> dof_indices(
                fe_values.dofs_per_cell);

              for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
                dof_indices[i] = cell->active_cell_index() - n_ghost_cells + i;

              std::vector<double> quadrature_values(
                fe_values.n_quadrature_points);
              fe_values.get_function_values(solution,
                                            dof_indices,
                                            quadrature_values);

              for (const unsigned int q_index :
                   fe_values.quadrature_point_indices())
                error += std::pow(quadrature_values[q_index] -
                                    exact_solution->value(
                                      fe_values.quadrature_point(q_index)),
                                  2.0) *
                         fe_values.JxW(q_index);
            }

        error = std::sqrt(error);
      }
  };

  // Perform time stepping
  DiscreteTime time(0, 1, delta_t);

  TimeStepping::MyExplicitRungeKutta<BlockVectorType> rk;
  rk.initialize(runge_kutta_method);

  postprocess(0.0, solution.block(0));

  while ((time.is_at_end() == false))
    {
      rk.evolve_one_time_step(fu_rhs,
                              time.get_current_time(),
                              time.get_next_step_size(),
                              solution);

      postprocess(time.get_current_time() + time.get_next_step_size(),
                  solution.block(0));

      time.advance_time();
    }

  table.add_value("n_cells", n_subdivisions_1D);
  table.add_value("degree", fe_degree);

  table.add_value("error", error);
  table.set_scientific("error", true);
  table.evaluate_convergence_rates(
    "error", "n_cells", ConvergenceTable::RateMode::reduction_rate_log2, dim);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const int          dim                    = 1;
  const unsigned int fe_degree              = 3;
  const unsigned int n_subdivisions_1D_min  = 10;
  const unsigned int n_subdivisions_1D_max  = 40;
  const unsigned int n_subdivisions_1D_step = 10;

  ConvergenceTable table;

  for (unsigned int n_subdivisions_1D = n_subdivisions_1D_min;
       n_subdivisions_1D <=
       std::max(n_subdivisions_1D_min, n_subdivisions_1D_max);
       n_subdivisions_1D += n_subdivisions_1D_step)
    {
      test<dim>(n_subdivisions_1D, fe_degree, table);
    }

  table.write_text(std::cout);
}