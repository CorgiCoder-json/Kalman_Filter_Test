//**************************************************************************************************
//
// Programmer: Jacob Maurer
// Description: Unscented Kalman Filter implementation in C++ using Eigen
//              Implementation I converted from:
//              https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/UKF.py
//              https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
//
//**************************************************************************************************

#include <iostream>
#include <chrono>
#include "UnscentedKalman.h"
#include "SystemSims.h"

template<int state_dim, int measure_dim>
UnscentedKalman<state_dim, measure_dim>::UnscentedKalman(
    std::function <Eigen::Vector<double, state_dim>(Eigen::Vector<double, state_dim>)> f,
    std::function <Eigen::Vector<double, measure_dim>(Eigen::Vector<double, state_dim>)> h,
    Eigen::Matrix<double, state_dim, state_dim> q,
    Eigen::Matrix<double, measure_dim, measure_dim> r,
    Eigen::Vector<double, state_dim> init_state,
    Eigen::Matrix<double, state_dim, state_dim> init_cov
    ) : predict_f(f), update_h(h), Q(q), R(r), state_vector(init_state), covariance_matrix(init_cov)
    {
        weights_m.fill(.5 / (state_dim + lambda_));
        weights_c.fill(.5 / (state_dim + lambda_));
        weights_c(0) = (lambda_ / (state_dim + lambda_)) + (1 - pow(alpha, 2) + beta);
        weights_m(0) = lambda_ / (state_dim + lambda_);
    };


template<int state_dim, int measure_dim>
void UnscentedKalman<state_dim, measure_dim>::unscented_transform_f(Eigen::Matrix<double, 2 * state_dim + 1, state_dim> sigmas)
{
    for (int i = 0; i < state_dim; i++)
    {
        state_vector_p[i] = weights_m.dot(sigmas.col(i));
    }
    Eigen::Matrix<double, state_dim, state_dim> temp = Eigen::Matrix<double, state_dim, state_dim>::Zero();
    covariance_matrix_p.setZero();
    for (int i = 0; i < sigma_dim; i++)
    {
        Eigen::Matrix<double, 1, state_dim> Y = sigmas.row(i) - state_vector_p.transpose();
        for (int j = 0; j < state_dim; j++)
        {
            temp.row(j) = (Y[j] * Y);
        }
        covariance_matrix_p += (weights_c[i] * temp);
    }
    covariance_matrix_p += Q;
}

template<int state_dim, int measure_dim>
void UnscentedKalman<state_dim, measure_dim>::unscented_transform_h(Eigen::Matrix<double, 2 * state_dim + 1, measure_dim> sigmas)
{

    for (int i = 0; i < measure_dim; i++)
    {
        unscented_x_z[i] = weights_m.dot(sigmas.col(i));
    }
    Eigen::Matrix<double, measure_dim, measure_dim> temp = Eigen::Matrix<double, measure_dim, measure_dim>::Zero();
    unscented_P_z.setZero();
    for (int i = 0; i < sigma_dim; i++)
    {
        Eigen::Matrix<double, 1, measure_dim> Y = sigmas.row(i) - unscented_x_z.transpose();
        for (int j = 0; j < measure_dim; j++)
        {
            temp.row(j) = (Y[j] * Y);
        }
        unscented_P_z += (weights_c[i] * temp);
    }
    unscented_P_z += R;
}

template<int state_dim, int measure_dim>
void UnscentedKalman<state_dim, measure_dim>::predict()
{
    Eigen::LLT<Eigen::Matrix<double, state_dim, state_dim>> U((state_dim + lambda_) * covariance_matrix);
    Eigen::Matrix<double, state_dim, state_dim> U_decomposed = U.matrixU();
    sigma_points.row(0) = state_vector.transpose();
    for (int i = 0; i < state_dim; i++)
    {
        sigma_points.row(i + 1) = state_vector.transpose() + U_decomposed.row(i);
        sigma_points.row(state_dim + i + 1) = state_vector.transpose() - U_decomposed.row(i);
    }
    for (int i = 0; i < sigma_dim; i++)
    {
        sigma_points_f.row(i) = this->predict_f(sigma_points.row(i));
    }
    this->unscented_transform_f(sigma_points_f);
}


template<int state_dim, int measure_dim>
void UnscentedKalman<state_dim, measure_dim>::update(const Eigen::Vector<double, measure_dim> measurements)
{
    sigma_points_h.setZero();
    for (int i = 0; i < sigma_dim; i++)
    {   
        sigma_points_h.row(i) = this->update_h(sigma_points_f.row(i));
    }
    this->unscented_transform_h(sigma_points_h);
    Eigen::Matrix<double, state_dim, measure_dim> covariance_x_z = Eigen::Matrix<double, state_dim, measure_dim>::Zero();
    Eigen::Matrix<double, state_dim, measure_dim> temp = Eigen::Matrix<double, state_dim, measure_dim>::Zero();
    for (int i = 0; i < sigma_dim; i++)
    {
        Eigen::Matrix<double, 1, measure_dim> inner1 = (sigma_points_h.row(i) - unscented_x_z.transpose());
        Eigen::Matrix<double, 1, state_dim> inner2 = (sigma_points_f.row(i) - state_vector_p.transpose());
        for (int j = 0; j < state_dim; j++)
        {
            temp.row(j) = inner2[j] * inner1;
        }
        covariance_x_z += (weights_c[i] * temp);
    }
    auto P_z_inv = unscented_P_z.inverse();
    auto K = covariance_x_z * P_z_inv;
    state_vector = state_vector_p + (K * (measurements - unscented_x_z)); 
    covariance_matrix = covariance_matrix_p - (K * unscented_P_z * K.transpose());
}


Eigen::Vector<double, 3> func_f(Eigen::Vector<double, 3> item)
{
    Eigen::Matrix<double, 3, 3> F;
    F << 1, 3.0, 0,
         0, 1, 0,
         0, 0, 1;
    return F * item;
}


Eigen::Vector<double, 2> func_h(Eigen::Vector<double, 3> item)
{
    double dx = item[0] - 0;
    double dy = item[2] - 0;
    double slant_range = sqrt(pow(dx, 2) + pow(dy, 2));
    double elevation_angle = atan2(dy, dx);
    return { slant_range, elevation_angle };
}

//Credit to: https://stackoverflow.com/questions/72096954/does-eigen-have-arange-function-like-numpy-arange-in-python
Eigen::VectorXd arange(double low, double high, double step, bool with_last = false)
{
    high -= (with_last) ? 0 : step;
    int N = static_cast<int>(std::floor((high - low) / step) + 1);
    return Eigen::VectorXd::LinSpaced(N, low, high);
}


int main()
{
    Eigen::Vector<double, 2> vel{ 100, 0 };
    Radar radar(0, 0, 5, 0.008726646259971648);
    ACSim sim(0, 1000, vel, 0.02);
    auto times = arange(0, 360 + 3, 3);
    Eigen::Vector<double, 3> x;
    Eigen::Matrix<double, 3, 3> P;
    Eigen::Matrix<double, 2, 2> R;

    x << 0, 90, 1100;
    P << pow(300, 2), 0, 0,
         0, pow(30, 2), 0,
         0, 0, pow(150,2);
    R << pow(5, 2), 0,
         0, pow(0.00872665, 2);

    UnscentedKalman<3, 2> filter(func_f, func_h, Eigen::Matrix<double, 3, 3>::Zero(), R, x, P);

    for (int i = 0; i < 1060; i++)
    {
        sim.update(3);
        auto r = radar.noisy_reading(sim.get_pos());
        filter.predict();
        filter.update(r);
    }
    std::cout << filter.get_state() << '\n';
    std::cout << filter.get_covar() << '\n';
    std::cout << sim.get_pos() << '\n';
}


