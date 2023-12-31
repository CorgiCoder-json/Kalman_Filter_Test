#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>

class Radar
{
public:
	Radar(double pos1, double pos2, double range_std, double angle_std);
	Eigen::Vector2d reading(Eigen::Vector2d ac_pos);
	Eigen::Vector2d noisy_reading(Eigen::Vector2d ac_pos);
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
	Eigen::Vector2d update(double dt);
	const Eigen::Vector2d get_pos() { return pos; }
private:
	Eigen::Vector2d pos;
	Eigen::Vector2d velocity;
	double vel_std_dev;
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<double> dist{0.0, 1.0};
};