#include <iostream>
#include <random>
#include "eig4.h"

bool jacobi_4x4(double * A, double * D, double * U)
{
    double B[4], Z[4];
    double Id[16] = {1., 0., 0., 0.,
                     0., 1., 0., 0.,
                     0., 0., 1., 0.,
                     0., 0., 0., 1.};

    memcpy(U, Id, 16 * sizeof(double));

    B[0] = A[0]; B[1] = A[5]; B[2] = A[10]; B[3] = A[15];
    memcpy(D, B, 4 * sizeof(double));
    memset(Z, 0, 4 * sizeof(double));

    for(int iter = 0; iter < 50; iter++) {
        double sum = fabs(A[1]) + fabs(A[2]) + fabs(A[3]) + fabs(A[6]) + fabs(A[7]) + fabs(A[11]);

        if (sum == 0.0)
            return true;

        double tresh =  (iter < 3) ? 0.2 * sum / 16. : 0.0;
        for(int i = 0; i < 3; i++) {
            double * pAij = A + 5 * i + 1;
            for(int j = i + 1 ; j < 4; j++) {
                double Aij = *pAij;
                double eps_machine = 100.0 * fabs(Aij);

                if ( iter > 3 && fabs(D[i]) + eps_machine == fabs(D[i]) && fabs(D[j]) + eps_machine == fabs(D[j]) )
                    *pAij = 0.0;
                else if (fabs(Aij) > tresh) {
                    double hh = D[j] - D[i], t;
                    if (fabs(hh) + eps_machine == fabs(hh))
                        t = Aij / hh;
                    else {
                        double theta = 0.5 * hh / Aij;
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if (theta < 0.0) t = -t;
                    }

                    hh = t * Aij;
                    Z[i] -= hh;
                    Z[j] += hh;
                    D[i] -= hh;
                    D[j] += hh;
                    *pAij = 0.0;

                    double c = 1.0 / sqrt(1 + t * t);
                    double s = t * c;
                    double tau = s / (1.0 + c);
                    for(int k = 0; k <= i - 1; k++) {
                        double g = A[k * 4 + i], h = A[k * 4 + j];
                        A[k * 4 + i] = g - s * (h + g * tau);
                        A[k * 4 + j] = h + s * (g - h * tau);
                    }
                    for(int k = i + 1; k <= j - 1; k++) {
                        double g = A[i * 4 + k], h = A[k * 4 + j];
                        A[i * 4 + k] = g - s * (h + g * tau);
                        A[k * 4 + j] = h + s * (g - h * tau);
                    }
                    for(int k = j + 1; k < 4; k++) {
                        double g = A[i * 4 + k], h = A[j * 4 + k];
                        A[i * 4 + k] = g - s * (h + g * tau);
                        A[j * 4 + k] = h + s * (g - h * tau);
                    }
                    for(int k = 0; k < 4; k++) {
                        double g = U[k * 4 + i], h = U[k * 4 + j];
                        U[k * 4 + i] = g - s * (h + g * tau);
                        U[k * 4 + j] = h + s * (g - h * tau);
                    }
                }
                pAij++;
            }
        }

        for(int i = 0; i < 4; i++) B[i] += Z[i];
        memcpy(D, B, 4 * sizeof(double));
        memset(Z, 0, 4 * sizeof(double));
    }

    return false;
}


int main() {
    const int numIterations = 1000000;
    const int matrixSize = 4;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1e6, 1e6);

    double A[matrixSize * matrixSize];
    double D[matrixSize];
    double U[matrixSize * matrixSize];

    {
        // Generate random symmetric matrix A
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j <= i; ++j) {
                double value = dis(gen);
                A[i * matrixSize + j] = value;
                A[j * matrixSize + i] = value;
            }
        }

        jacobi_4x4(A, D, U);

        Eigen::Vector4d DD(D);
        std::cout << "Eigenvalues: " << std::endl << DD.transpose() << std::endl;

        Eigen::Matrix4d UU(U);
        std::cout << "Eigenvectors: " << std::endl << UU << std::endl;

        Eigen::Vector4d vec;
        double val;
        Eigen::Matrix4d AA(A);
        eig4(AA, vec, val);
        std::cout << "Eigenvalue: " << val << std::endl;
        std::cout << "Eigenvectors: " << std::endl << vec.transpose() << std::endl;
    }

    // Benchmark jacobi_4x4
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < numIterations; ++iter) {
        // Generate random symmetric matrix A
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j <= i; ++j) {
                double value = dis(gen);
                A[i * matrixSize + j] = value;
                A[j * matrixSize + i] = value;
            }
        }

        jacobi_4x4(A, D, U);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "jacobi_4x4 execution time: " << duration.count() / numIterations << " ns" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < numIterations; ++iter) {
        // Generate random symmetric matrix A
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j <= i; ++j) {
                double value = dis(gen);
                A[i * matrixSize + j] = value;
                A[j * matrixSize + i] = value;
            }
        }

        Eigen::Vector4d vec;
        double val;
        Eigen::Matrix4d AA(A);
        eig4(AA, vec, val);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "eig4 execution time: " << duration.count() / numIterations << " ns" << std::endl;


    return 0;
}