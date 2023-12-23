#include "SystemSims.h"

Radar::Radar(double pos1, double pos2, double range_std, double angle_std) {
	this->pos = { pos1, pos2 };
	this->range_std_dev = range_std;
	this->elev_angle_std_dev = angle_std;
}

Eigen::Vector2d Radar::reading(Eigen::Vector2d ac_pos)
{
	Eigen::Vector2d diff = (ac_pos - this->pos);
	double rng = diff.norm();
	double brg = atan2(diff[1], diff[0]);
	return { rng, brg };
}

Eigen::Vector2d Radar::noisy_reading(Eigen::Vector2d ac_pos)
{
	Eigen::Vector2d read = this->reading(ac_pos);
	Eigen::Vector2d addend = {dist(gen) * range_std_dev, dist(gen) * elev_angle_std_dev};
	return addend + read;
}

ACSim::ACSim(double pos1, double pos2, Eigen::Vector2d vel, double vel_std)
{
	this->pos = { pos1, pos2 };
	this->velocity = vel;
	this->vel_std_dev = vel_std;
}

Eigen::Vector2d ACSim::update(double dt) 
{
	Eigen::Vector2d dx = velocity * dt;
	return pos + dx;
}
