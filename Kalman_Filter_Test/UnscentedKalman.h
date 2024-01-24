//**************************************************************************************************
//
// Programmer: Jacob Maurer
// Date: 1/23/2024
// Description: Unscented Kalman Filter declarations in C++ using Eigen
//              Implementation I converted from:
//              https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/UKF.py
//              https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
//
//**************************************************************************************************
#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Cholesky/LLT.h>
#include <Eigen/src/Core/Matrix.h>

template<int state_dim, int measure_dim>
class UnscentedKalman
{
public:
    UnscentedKalman(std::function <Eigen::Vector<double, state_dim>(Eigen::Vector<double, state_dim>)> f,
        std::function <Eigen::Vector<double, measure_dim>(Eigen::Vector<double, state_dim>)> h,
        Eigen::Matrix<double, state_dim, state_dim> q,
        Eigen::Matrix<double, measure_dim, measure_dim> r,
        Eigen::Vector<double, state_dim> init_state,
        Eigen::Matrix<double, state_dim, state_dim> init_cov);
    void predict();
    void update(const Eigen::Vector<double, measure_dim> measurements);
    void unscented_transform_f(Eigen::Matrix<double, 2 * state_dim + 1, state_dim> sigmas); //Add spot for Q and R matricies
    void unscented_transform_h(Eigen::Matrix<double, 2 * state_dim + 1, measure_dim> sigmas);
    const Eigen::Vector<double, state_dim> get_state() { return state_vector; }
    const Eigen::Matrix<double, state_dim, state_dim> get_covar() { return covariance_matrix; }
    const Eigen::Vector<double, state_dim> get_state_p() { return state_vector_p; }
    const Eigen::Matrix<double, state_dim, state_dim> get_covar_p() { return covariance_matrix_p; }
private:
    static const int sigma_dim = 2 * state_dim + 1;
    const double kappa = 0;
    const double alpha = .1;
    const double beta = 2.0;
    const double lambda_ = pow(alpha, 2) * (state_dim + kappa) - state_dim;
    std::function <Eigen::Vector<double, state_dim>(Eigen::Vector<double, state_dim>)> predict_f;
    std::function <Eigen::Vector<double, measure_dim>(Eigen::Vector<double, state_dim>)> update_h;
    Eigen::Matrix<double, 1, sigma_dim> weights_m;
    Eigen::Matrix<double, 1, sigma_dim> weights_c;
    Eigen::Vector<double, state_dim> state_vector;
    Eigen::Matrix<double, state_dim, state_dim> covariance_matrix;
    Eigen::Vector<double, state_dim> state_vector_p = Eigen::Vector<double, state_dim>::Zero();
    Eigen::Matrix<double, state_dim, state_dim> covariance_matrix_p = Eigen::Matrix<double, state_dim, state_dim>::Identity();
    Eigen::Matrix<double, sigma_dim, state_dim> sigma_points = Eigen::Matrix<double, sigma_dim, state_dim>::Ones();
    Eigen::Matrix<double, sigma_dim, state_dim> sigma_points_f = Eigen::Matrix<double, sigma_dim, state_dim>::Ones();
    Eigen::Matrix<double, sigma_dim, measure_dim> sigma_points_h = Eigen::Matrix<double, sigma_dim, measure_dim>::Ones();
    Eigen::Vector<double, measure_dim> unscented_x_z = Eigen::Vector<double, measure_dim>::Ones();
    Eigen::Matrix<double, measure_dim, measure_dim> unscented_P_z = Eigen::Matrix<double, measure_dim, measure_dim>::Zero();
    Eigen::Matrix<double, state_dim, state_dim> Q;
    Eigen::Matrix<double, measure_dim, measure_dim> R;

};
