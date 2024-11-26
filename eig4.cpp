// Author: Jin Wu
// E-mail: jin_wu_uestc@hotmail.com
// Publication: Wu, J., et al. (2020) Fast Symbolic 3D Registration Solution.
//                      IEEE Transactions on Automation Science and Engineering.
// This code file includes the eig4 function for computing the mininum
// eigenvalue with corresponding eigenvector.
// For computing the maximum eigenvalue, you may replace input A with -A.


#include "eig4.h"


void eig4(const Eigen::Matrix4d& A, Eigen::Vector4d &v, double &lambda) {
    double a11 = A(0, 0), a12 = A(0, 1), a13 = A(0, 2), a14 = A(0, 3);
    double a22 = A(1, 1), a23 = A(1, 2), a24 = A(1, 3);
    double a33 = A(2, 2), a34 = A(2, 3);
    double a44 = A(3, 3);

    double a12_2 = a12 * a12, a13_2 = a13 * a13, a14_2 = a14 * a14;
    double a23_2 = a23 * a23, a24_2 = a24 * a24;
    double a34_2 = a34 * a34;

    double tau1 = -a11 - a22 - a33 - a44;
    double tau2 = -a12_2 - a13_2 - a14_2 + a11 * a22 - a23_2 - a24_2 + a11 * a33 + a22 * a33 - a34_2 + a11 * a44 + a22 * a44 + a33 * a44;
    double tau3 = a13_2 * a22 + a14_2 * a22 - 2 * a12 * a13 * a23 + a11 * a23_2 -
                  2 * a12 * a14 * a24 + a11 * a24_2 + a12_2 * a33 + a14_2 * a33 -
                  a11 * a22 * a33 + a24_2 * a33 - 2 * a13 * a14 * a34 - 2 * a23 * a24 * a34 +
                  a11 * a34_2 + a22 * a34_2 + a12_2 * a44 + a13_2 * a44 -
                  a11 * a22 * a44 + a23_2 * a44 - a11 * a33 * a44 - a22 * a33 * a44;
    double tau4 = a14_2 * a23_2 - 2 * a13 * a14 * a23 * a24 + a13_2 * a24_2 -
                  a14_2 * a22 * a33 + 2 * a12 * a14 * a24 * a33 - a11 * a24_2 * a33 + 2 * a13 * a14 * a22 * a34 -
                  2 * a12 * a14 * a23 * a34 - 2 * a12 * a13 * a24 * a34 + 2 * a11 * a23 * a24 * a34 +
                  a12_2 * a34_2 - a11 * a22 * a34_2 - a13_2 * a22 * a44 +
                  2 * a12 * a13 * a23 * a44 - a11 * a23_2 * a44 - a12_2 * a33 * a44 + a11 * a22 * a33 * a44;

    double tau1_2 = tau1 * tau1;
    double a = tau2 - 3.0 / 8.0 * tau1_2;
    double b = tau3 - tau1 * tau2 / 2.0 + tau1_2 * tau1 / 8.0;
    double c = tau4 - tau1 * tau3 / 4.0 + tau1_2 * tau2 / 16.0 - 3.0 * tau1_2 * tau1_2 / 256.0;

    double T0 = 2.0 * std::pow(a, 3) + 27.0 * b * b - 72.0 * a * c;
    double theta = std::atan2(std::sqrt(4.0 * std::pow(a * a + 12.0 * c, 3) - T0 * T0), T0);
    double aT1 = 1.259921049894873 * std::sqrt(a * a + 12.0 * c) * std::cos(theta / 3.0);
    double T2 = std::sqrt(-4.0 * a + 3.174802103936399 * aT1);

    lambda = -0.204124145231932 * (T2 + std::sqrt(-T2 * T2 - 12.0 * a + 29.393876913398135 * b / T2)) - tau1 / 4.0;

    double G11 = A(0, 0) - lambda, G12 = A(0, 1), G13 = A(0, 2), G14 = A(0, 3);
    double G22 = A(1, 1) - lambda, G23 = A(1, 2), G24 = A(1, 3);
    double G33 = A(2, 2) - lambda, G34 = A(2, 3);

    v << G14 * G23 * G23 - G13 * G23 * G24 - G14 * G22 * G33 + G12 * G24 * G33 + G13 * G22 * G34 - G12 * G23 * G34,
            G13 * G13 * G24 + G12 * G14 * G33 - G11 * G24 * G33 + G11 * G23 * G34 - G13 * G14 * G23 - G13 * G12 * G34,
            G13 * G14 * G22 - G12 * G14 * G23 - G12 * G13 * G24 + G11 * G23 * G24 + G12 * G12 * G34 - G11 * G22 * G34,
            -(G13 * G13 * G22 - 2 * G12 * G13 * G23 + G11 * G23 * G23 + G12 * G12 * G33 - G11 * G22 * G33);

    v = v.normalized();
}