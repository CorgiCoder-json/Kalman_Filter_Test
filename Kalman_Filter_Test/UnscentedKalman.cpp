#include "UnscentedKalman.h"

template<int state_dim, int measure_dim>
void UnscentedKalman<state_dim, measure_dim>::unscented_transform(Eigen::Matrix<double, 2 * state_dim + 1, state_dim> sigmas)
{
    unscented_x = weights_m.dot(sigmas);
    for (int i = 0; i < sigma_dim; i++)
    {
        auto Y = sigma_points_f[i] - state_vector;
        unscented_P += weights_c[i] * (Y * Y.transpose());
    }
    //unscented_P += Parameter Q or R;
}

template<int state_dim, int measure_dim>
void UnscentedKalman<state_dim, measure_dim>::predict()
{
    this->weights_m.fill(1 / (2 * (state_dim + lambda_)));
    this->weights_c.fill(1 / (2 * (state_dim + lambda_)));
    this->weights_m[0] = lambda_ / (state_dim + lambda_) + (1 - pow(alpha, 2) + beta);
    this->weights_c[0] = lambda_ / (state_dim + lambda_);
    Eigen::LLT<Eigen::Matrix<double, state_dim, state_dim>> U((state_dim + lambda_) * covariance_matrix);
    Eigen::Matrix<double, state_dim, state_dim> U_decomposed = U.matrixU();
    sigma_points.col(0) << state_vector;
    for (int i = 0; i < state_dim; i++)
    {
        sigma_points[i + 1] = state_vector + U_decomposed[i];
        sigma_points[state_dim + i + 1] = state_vector - U_decomposed[i];
    }
    for (int i = 0; i < sigma_dim; i++)
    {
        sigma_points_f[i] = this->predict_function_f(sigma_points[i], .02);
    }
    this->unscented_transform(sigma_points_f);
    state_vector_p = unscented_x;
    covariance_matrix_p = unscented_P;
}

template<int state_dim, int measure_dim>
void UnscentedKalman<state_dim, measure_dim>::update(const Eigen::Vector<double, measure_dim>& measurements)
{
    for (int i = 0; i < sigma_dim; i++)
    {
        sigma_points_h[i] = this->update_function_h(sigma_points_f[i]);
    }
    this->unscented_transform(sigma_points_h);
    Eigen::Matrix<double, state_dim, measure_dim> covariance_x_z;
    for (int i = 0; i < sigma_dim; i++)
    {
        covariance_x_z += weights_c[i] * ((sigma_points_f[i] - state_vector) * (sigma_points_h[i] - unscented_x).transpose());
    }
    auto K = covariance_x_z * unscented_P.inverse();
    state_vector = state_vector_p + K.dot(measurements - unscented_x);
    covariance_matrix = covariance_matrix_p + K.dot(unscented_P).dot(K.transpose());
}

template<int state_dim, int measure_dim>
Eigen::Vector<double, state_dim> UnscentedKalman<state_dim, measure_dim>::predict_function_f()
{
    Eigen::Matrix<double, state_dim, state_dim> F = { {1, .2, 0},
                                                      {0, 1, 0},
                                                      {0, 0, 1} };
    return F * state_vector;
}

template<int state_dim, int measure_dim>
Eigen::Vector<double, measure_dim> UnscentedKalman<state_dim, measure_dim>::update_function_h()
{
    double dx = state_vector[0];
    double dy = state_vector[2];
    double slant_range = sqrtf(pow(dx, 2) + pow(dy, 2));
    double elevation_angle = atan2f(dy, dx);
    return { slant_range, elevation_angle };
}

