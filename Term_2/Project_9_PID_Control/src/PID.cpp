#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Tau_p, double Tau_i, double Tau_d) 
{
	// P: steer in proportion to the crosstrack error

	// I: steer more when there is sustained error to counter the systematic bias we have from e.g. misaligned wheels.

	// D: When the car has turned enough to reduce CTE, it counter-steers
	// to avoid overshooting

	// Twiddle: choose optimal parameters

	Kp = Tau_p;
	Ki= Tau_i;
	Kd = Tau_d;
	total_cte = 0;
	last_cte = 0;
}

void PID::UpdateError(double cte) 
{
	total_cte += cte;
	p_error = -Kp * cte;
	i_error = -Ki * total_cte;
	d_error = -Kd * (cte - last_cte);
	last_cte = cte;
}

double PID::TotalError() 
{

	return (p_error + i_error + d_error);
}