#pragma once

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
