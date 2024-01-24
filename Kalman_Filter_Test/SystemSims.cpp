//**************************************************************************************************
//
// Programmer: Jacob Maurer
// Date: 1/23/2024
// Description: System Sim implementation in C++ using Eigen
//              Implementation I converted from:
//              https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
//
//**************************************************************************************************
#include "SystemSims.h"

//Creates a simulated radar sensor
Radar::Radar(double pos1, double pos2, double range_std, double angle_std) {
	this->pos = { pos1, pos2 };
	this->range_std_dev = range_std;
	this->elev_angle_std_dev = angle_std;
}

//provides and actual reading with no noise
Eigen::Vector2d Radar::reading(Eigen::Vector2d ac_pos)
{
	Eigen::Vector2d diff = (ac_pos - this->pos);
	double rng = diff.norm();
	double brg = atan2(diff[1], diff[0]);
	return { rng, brg };
}

//provides noise to a reading
Eigen::Vector2d Radar::noisy_reading(Eigen::Vector2d ac_pos)
{
	Eigen::Vector2d read = this->reading(ac_pos);
	Eigen::Vector2d addend = {dist(gen) * range_std_dev, dist(gen) * elev_angle_std_dev};
	return (addend + read);
}

//Constructor that simulates the aircraft
ACSim::ACSim(double pos1, double pos2, Eigen::Vector2d vel, double vel_std)
{
	this->pos = { pos1, pos2 };
	this->velocity = vel;
	this->vel_std_dev = vel_std;
}

//updates the simulated aircraft's actual positon
void ACSim::update(double dt) 
{
	Eigen::Vector2d add = { (dist(gen) * vel_std_dev) * dt, (dist(gen) * vel_std_dev) * dt };
	Eigen::Vector2d dx = (velocity * dt) + add;
	this->pos += dx;
}
