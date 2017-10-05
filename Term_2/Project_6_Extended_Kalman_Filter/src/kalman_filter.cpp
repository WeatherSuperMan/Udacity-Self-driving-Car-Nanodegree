#include "kalman_filter.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;


const float PI2 = 2 * M_PI;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
	MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
	x_ = x_in;
	P_ = P_in;
	F_ = F_in;
	H_ = H_in;
	R_ = R_in;
	Q_ = Q_in;
}

void KalmanFilter::Predict() {
	/**
	TODO:
	* predict the state
	*/

	x_ = F_*x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	/**
	TODO:
	* update the state by using Kalman Filter equations
	*/

	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd H_transposed = H_.transpose();
	MatrixXd S = H_ * P_ * H_transposed + R_;
	MatrixXd S_inversed = S.inverse();
	MatrixXd K = P_ * H_transposed * S_inversed;

	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	/**
	TODO:
	* update the state by using Extended Kalman Filter equations
	*/

	float px = x_[0];
	float py = x_[1];
	float vx = x_[2];
	float vy = x_[3];

	float rho = sqrt(px*px + py*py);
	float phi = atan2(py, px);


	// if rho is very small, set it to 0.0001 to avoid division by 0 in computing rho_dot
	if (rho < 0.000001)
		rho = 0.000001;

	float rho_dot = (px*vx + py*vy) / rho;

	// Calculate h(x_p)
	VectorXd Hx_(3);
	Hx_ << rho, phi, rho_dot;

	// Calculate y
	VectorXd y = z - Hx_;

	// Adjust phi for y

	while (y[1] > M_PI)
	{
		y[1] -= PI2;
	}

	while (y[1] < -M_PI)
	{
		y[1] += PI2;
	}

	//TODO Might be better to create a function for below

	MatrixXd H_transposed = H_.transpose();
	MatrixXd S = H_*P_*H_transposed + R_;
	MatrixXd S_inversed = S.inverse();
	MatrixXd K = P_*H_transposed*S_inversed;

	x_ = x_ + (K*y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K*H_)*P_;

}