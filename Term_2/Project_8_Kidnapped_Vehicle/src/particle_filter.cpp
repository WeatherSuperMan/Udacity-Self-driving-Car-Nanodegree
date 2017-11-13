/*
* particle_filter.cpp
*
*  Created on: Dec 12, 2016
*     Original Author: Tiffany Huang
      Modified by:  John Wu
	  Credit: Sahil Juneja
*/

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	int num_particles = 300;

	// Generate noise
	default_random_engine gen;

	normal_distribution<double> x_init(0, std[0]);
	normal_distribution<double> y_init(0, std[1]);
	normal_distribution<double> theta_init(0, std[2]);

	//vector<double> weights; // simplify resample

	//vector<Particle> particles(num_particles);
	for (int i = 0; i < num_particles; i++)
	{
		Particle new_particle;
		new_particle.id = i;
		new_particle.x = x + x_init(gen);
		new_particle.y = y + y_init(gen);
		new_particle.theta = theta + theta_init(gen);
		new_particle.weight = 1.0;
		
		// Add particle to list of particles
		particles.push_back(new_particle);
		weights.push_back(1.0);

	}

	
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Generate noise
	default_random_engine gen;

	normal_distribution<double> x_init(0, std_pos[0]);
	normal_distribution<double> y_init(0, std_pos[1]);
	normal_distribution<double> theta_init(0, std_pos[2]);


	for (int i = 0; i < particles.size(); i++)
	{

		// what if yaw_rate is 0
		if (fabs(yaw_rate) < 0.00001)
		{
			particles[i].x += velocity*delta_t*cos(particles[i].theta);
			particles[i].y += velocity*delta_t*sin(particles[i].theta);
			particles[i].theta = particles[i].theta;

		}
		else
		{
			particles[i].x += velocity*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) / yaw_rate;
			particles[i].y += velocity*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) / yaw_rate;

			particles[i].theta += yaw_rate*delta_t;
		}


		// add noise
		particles[i].x += x_init(gen);
		particles[i].y += y_init(gen);
		particles[i].theta += theta_init(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++)

	{
		// current distance to be initialised to its maximum possible value
		double current_dist = numeric_limits<double>::max();

		int map_id;

		for (int j = 0; j < predicted.size(); j++)
		{
			// calculate the distance between current and predicted landmarks.
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);


			// applying nearest-neighbor technique, to find the nearest predicted landmark for each observation
			if (distance < current_dist)
			{
				current_dist = distance;
				map_id = predicted[j].id;
			}
		}

		//set the observation's id to the nearest predicted landmark's id to complete landmark association
		observations[i].id = map_id;

	}

}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	double var_x = std_landmark[0] * std_landmark[0];
	double var_y = std_landmark[1] * std_landmark[1];
	double convar_xy = std_landmark[0] * std_landmark[1];


	for (int i = 0; i < particles.size(); i++)
	{
		// Transform Observations to Map Coordinates
		//vector<LandmarkObs> obs_transformed(observations.size()); 
																  
		vector<LandmarkObs> obs_transformed;

		for (int j = 0; j < observations.size(); j++)
		{
			/*
			obs_transformed[j].x = particles[i].x + observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta);
			obs_transformed[j].y = particles[i].y + observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta);
			obs_transformed[j].id = observations[j].id;
			*/

			double t_x = particles[i].x + observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta);
			double t_y = particles[i].y + observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta);
			obs_transformed.push_back(LandmarkObs{ observations[j].id, t_x, t_y });


		}


		// Find nearest map landmark
		vector<LandmarkObs> closest_landmark;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			

			double current_distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, 
				map_landmarks.landmark_list[j].y_f);

			if (current_distance < sensor_range)
			{
			
				LandmarkObs predicted_landmark;
				predicted_landmark.x = map_landmarks.landmark_list[j].x_f;
				predicted_landmark.y = map_landmarks.landmark_list[j].y_f;
				predicted_landmark.id = map_landmarks.landmark_list[j].id_i;

				closest_landmark.push_back(predicted_landmark);

			}

		}



		// Call data association function
		dataAssociation(closest_landmark, obs_transformed);


		// Update weights
		double initial_weight = 1.0;
	
	

		for (int k = 0; k < obs_transformed.size(); k++)
		{
			double x_predicted, y_predicted;


			for (int j = 0; j < closest_landmark.size(); j++)
			{
				if (closest_landmark[j].id == obs_transformed[k].id)
				{
					x_predicted = closest_landmark[j].x;
					y_predicted = closest_landmark[j].y;
				}
			}

			// calculate the multivariate Gaussian Probabilities
			double coefficient = 1 / (2 * M_PI*convar_xy);
			double exponent = exp(-((pow((obs_transformed[k].x - x_predicted), 2) / (2 * var_x)) +
				(pow((obs_transformed[k].y - y_predicted), 2) / (2 * var_y))));

			initial_weight *= coefficient * exponent; // take the product of all weights

			

		}

		particles[i].weight = initial_weight;
		weights[i] = initial_weight;

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> discrete_dist(weights.begin(), weights.end());

	vector<Particle> resample_particles(particles.size());

	int rand_num = discrete_dist(gen);

	for (int A = 0; A < particles.size(); A++)
	{
		resample_particles[A] = particles[rand_num];
		weights[A] = particles[rand_num].weight;

	}

	// resample complete
	particles = resample_particles;

}



Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}


string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}


string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}


string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}