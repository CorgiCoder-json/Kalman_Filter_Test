//**************************************************************************************************
//
// Programmer: Jacob Maurer
// Date: 1/23/2024
// Description: System Sim declarations in C++ using Eigen
//              Implementation I converted from:
//              https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
//
//**************************************************************************************************

#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>

class Radar
{
public:
	Radar(double pos1, double pos2, double range_std, double angle_std);
	Eigen::Vector<double, 2> reading(Eigen::Vector2d ac_pos);
	Eigen::Vector<double, 2> noisy_reading(Eigen::Vector2d ac_pos);
private:
	Eigen::Vector2d pos;
	double range_std_dev;
	double elev_angle_std_dev;
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<double> dist{0.0, 1.0};
};

class ACSim
{
public:
	ACSim(double pos1, double pos2, Eigen::Vector2d vel, double vel_std);
	void update(double dt);
	const Eigen::Vector2d get_pos() { return pos; }
private:
	Eigen::Vector<double, 2> pos;
	Eigen::Vector<double, 2> velocity;
	double vel_std_dev;
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<double> dist{0.0, 1.0};
};