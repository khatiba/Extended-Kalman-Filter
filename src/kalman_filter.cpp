#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
  // initialize state vector x
  x_ = VectorXd(4);
  x_ << 1, 1, 1, 1;

  // the initial transition matrix F_ without dt
  F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

  // state covariance matrix P
  P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0   , 0,
        0, 1, 0   , 0,
        0, 0, 1000, 0,
        0, 0, 0   , 1000;

  Q_ = MatrixXd(4, 4);
  Q_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1;

  long x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  Update_(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float rho       = sqrt(px*px + py*py);
  float phi       = atan2(py, px);

  // don't divide by zero
  if (rho < 0.0001) {
    rho = 0.0001;
  }
  
  float rho_dot   = (px*vx + py*vy)/rho;

  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;
  VectorXd y = z - z_pred;

  // normalize delta-rho between -pi, pi
  while (y(1) > M_PI)
    y(1) -= 2.0 * M_PI;
  while (y(1) < -M_PI)
    y(1) += 2.0 * M_PI;

  Update_(y);
}

void KalmanFilter::Update_(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;

  // new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}

