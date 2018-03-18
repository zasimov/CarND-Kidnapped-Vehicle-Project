/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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
#include <cassert>

#include "particle_filter.h"

using namespace std;

/*
 * Particle
 */
void Particle::Init(const int new_id, const double sample_x, const double sample_y, const double sample_theta, const double init_weight) {
  id = new_id;
  x = sample_x;
  y = sample_y;
  theta = sample_theta;
  weight = init_weight;
}


void Particle::Update(const double dt,
		      const double velocity, const double yawrate,
		      const double noise_x, const double noise_y, const double noise_theta) {
  if (IsZero(yawrate)) {
    x += velocity * dt * cos(theta);
    y += velocity * dt * sin(theta);
  } else {
    const double theta_dt = yawrate * dt;
    const double new_theta = theta + theta_dt;
    const double v = velocity / yawrate;
    x += v * (sin(new_theta) - sin(theta));
    y += v * (cos(theta) - cos(new_theta));
    theta += theta_dt;
  }

  // add noise
  x += noise_x;
  y += noise_y;
  theta += noise_theta;
}


std::vector<double> Particle::ToMapCoords(const double ox, const double oy) const {
  std::vector<double> map_coords(2);

  map_coords[0 /* x */] = x + ox * cos(theta) - oy * sin(theta);
  map_coords[1 /* y */] = y + ox * sin(theta) + oy * cos(theta);

  return map_coords;
}


double Particle::DistanceTo(const double ox, const double oy) const {
  return dist(x, y, ox, oy);
}

/*
 * ParticleFilter
 */

void ParticleFilter::init(double x, double y, double theta, double std[3]) {
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (unsigned i = 0; i < num_particles; i++) {
    const double sample_x = dist_x(gen_);
    const double sample_y = dist_y(gen_);
    const double sample_theta = dist_theta(gen_);

    particles[i].Init(i, sample_x, sample_y, sample_theta, 1.0);
    weights[i] = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[3], double velocity, double yaw_rate) {
  std::normal_distribution<double> dist_noise_x(0, std_pos[0]);
  std::normal_distribution<double> dist_noise_y(0, std_pos[1]);
  std::normal_distribution<double> dist_noise_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    particles[i].Update(delta_t, velocity, yaw_rate,
			dist_noise_x(gen_), dist_noise_y(gen_), dist_noise_theta(gen_));
  }
}

/*
 * Convert observations from vehicle coords system to map coords system.
 *
 *   observations -> map_observations
 *
 */
static void ToMapCoords(const Particle &particle, const std::vector<LandmarkObs> observations, std::vector<LandmarkObs> &map_observations) {
  assert(observations.size() == map_observations.size());

  for (unsigned i = 0; i < observations.size(); i++) {
    map_observations[i].id = observations[i].id;

    const std::vector<double> mapped = particle.ToMapCoords(observations[i].x, observations[i].y);
    map_observations[i].x = mapped[0];
    map_observations[i].y = mapped[1];
  }
}

void ParticleFilter::findNearestNeighbor(std::vector<LandmarkObs>& observations,
					 const std::vector<LandmarkObs> &visible_landmarks) {
  assert(visible_landmarks.size() != 0);

  for (unsigned i = 0; i < observations.size(); i++) {
    unsigned min_visible_landmark_idx = 0;
    double min_distance = dist(observations[i].x, observations[i].y,
			       visible_landmarks[0].x, visible_landmarks[0].y);

    for (unsigned j = 1; j < visible_landmarks.size(); j++) {
      // cur_distance is a distance from observation[i] to visible_landmark[j]
      const double cur_distance = dist(observations[i].x, observations[i].y,
				       visible_landmarks[j].x, visible_landmarks[j].y);

      if (cur_distance < min_distance) {
	min_distance = cur_distance;
	min_visible_landmark_idx = j;
      }
    }

    observations[i].id = min_visible_landmark_idx;
  }
}

/*
 * Multivariate Gaussian Distribution (in 2D)
 */
double mgd2d(const LandmarkObs &obs, const LandmarkObs &pred, const double std_landmark[2]) {
    const double cov2_x = 2 * std_landmark[0] * std_landmark[0];
    const double cov2_y = 2 * std_landmark[1] * std_landmark[1];
    const double gauss_norm = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
    const double dx = obs.x - pred.x;
    const double dy = obs.y - pred.y;
    const double exponent = dx * dx / cov2_x + dy * dy / cov2_y;
    return exp(-exponent) / gauss_norm;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[2],
				   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  update_cycle_num_++;

  // observations in map coords
  std::vector<LandmarkObs> map_observations(observations.size());

  for (unsigned i = 0; i < num_particles; i++) {
    auto &particle = particles[i];

    ToMapCoords(particle, observations, map_observations);

    // find "visible" landmarks
    std::vector<LandmarkObs> visible_landmarks;
    for (unsigned j = 0; j < map_landmarks.landmark_list.size(); j++) {
      const auto &lm = map_landmarks.landmark_list[j];

      if (particle.DistanceTo(lm.x_f, lm.y_f) < sensor_range) {
	visible_landmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }

      /*
      if ( (fabs(lm.x_f - particle.x) <= sensor_range) and (fabs(lm.y_f - particle.y) <= sensor_range) ) {
	visible_landmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
      */
    }

    if (visible_landmarks.size() == 0) {
      std::cout << "warning: visible_landmarks is empty (" << particle.x << ", " << particle.y << ")" << std::endl;
      continue;
    }

    // After this step observation.id (observation in
    // map_observations) contains index of visible_landmark.
    findNearestNeighbor(map_observations, visible_landmarks);

    // update particle weight
    double weight = 1.0;

    for (unsigned j = 0; j < map_observations.size(); j++) {
      const auto &obs = map_observations[j];
      const auto &pred = visible_landmarks[obs.id];

      weight *= mgd2d(obs, pred, std_landmark);
    }

    particle.weight = weight;
    weights[i] = weight;

    journal_->Write(update_cycle_num_, particle.id, particle.x, particle.y, particle.theta, weight, observations.size(), visible_landmarks.size());
  }
}

void ParticleFilter::resample() {
  std::vector<Particle> new_particles;
  std::vector<double> new_weights;

  uniform_int_distribution<int> uni(0, num_particles-1);
  auto index = uni(gen_);

  double mw = *std::max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> wdist(0.0, 2.0 * mw);

  double beta = 0.0;

  for (int i = 0; i < num_particles; i++) {
    beta += wdist(gen_);
    while (weights[index] < beta) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
    new_weights.push_back(particles[index].weight);
  }

  particles = new_particles;
  weights = new_weights;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
					 const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
