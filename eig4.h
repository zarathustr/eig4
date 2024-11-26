#ifndef EIG4_EIG4_H
#define EIG4_EIG4_H

#include <chrono>
#include <Eigen/Dense>

void eig4(const Eigen::Matrix4d& A, Eigen::Vector4d &v, double &lambda);

#endif //EIG4_EIG4_H
