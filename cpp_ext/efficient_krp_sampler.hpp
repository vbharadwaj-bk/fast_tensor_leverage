#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include "common.h"
#include "sampler.hpp"
#include "partition_tree.hpp"
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

class __attribute__((visibility("hidden"))) EfficientKRPSampler: public Sampler {
public:
    ScratchBuffer scratch;
    Buffer<double> M;
    Buffer<double> lambda;

    Buffer<double> scaled_h;
    vector<Buffer<double>> scaled_eigenvecs;

    vector<PartitionTree*> gram_trees;
    vector<PartitionTree*> eigen_trees;
    double eigenvalue_tolerance;

    // Related to random number generation 
    std::uniform_real_distribution<> dis;

    EfficientKRPSampler(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :       
            Sampler(J, R, U_matrices),
            scratch(R, J, R),
            M({U_matrices.size() + 2, R * R}),
            lambda({U_matrices.size() + 1, R}),
            scaled_h({J, R}),
            dis(0.0, 1.0) 
    {    
        eigenvalue_tolerance = 1e-8; // Tolerance of eigenvalues for symmetric PINV 
    
        for(uint32_t i = 0; i < N; i++) {
            uint32_t n = U[i].shape[0];
            assert(U[i].shape.size() == 2);
            assert(U[i].shape[1] == R);
            //assert(n % R == 0);  // Should check these assertions outside this class!

            uint64_t F = R < n ? R : n;
            gram_trees.push_back(new PartitionTree(n, F, J, R, scratch));
            eigen_trees.push_back(new PartitionTree(R, 1, J, R, scratch));
        }

        // Should move the data structure initialization to another routine,
        // but this is fine for now.

        for(uint32_t i = 0; i < N; i++) {
            gram_trees[i]->build_tree(U[i]); 
        }

        for(uint32_t i = 0; i < N + 1; i++) {
            scaled_eigenvecs.emplace_back(initializer_list<uint64_t>{R, R}, M(i, 0));
        }
    }

    /*
    * Updates the j'th gram tree when the factor matrix is
    * updated. 
    */
    void update_sampler(uint64_t j) {
        gram_trees[j]->build_tree(U[j]); 
    }

    /*
     * Simple, unoptimized square-matrix in-place transpose.
    */
    void transpose_square_in_place(double* ptr, uint64_t n) {
        for(uint64_t i = 0; i < n - 1; i++) {
            for(uint64_t j = i + 1; j < n; j++) {
                double temp = ptr[i * n + j];
                ptr[i * n + j] = ptr[j * n + i];
                ptr[j * n + i] = temp;
            }
        }
    }

    void computeM(uint32_t j) {
        std::fill(M(N * R2), M((N + 1) * R2), 1.0);

        #pragma omp parallel
{
        uint32_t last_buffer = N;
        for(int k = N - 1; k >= 0; k--) {
            if((uint32_t) k != j) {
                #pragma omp for
                for(uint32_t i = 0; i < R2; i++) {
                    M[k * R2 + i] = gram_trees[k]->G[i] * M[(last_buffer * R2) + i];   
                } 

                last_buffer = k;
            }
        }
}

        if(j == 0) {
            std::copy(M(1, 0), M(1, R2), M());
        }

        // Store the originla matrix in slot N + 2 
        std::copy(M(), M(R2), M((N + 1) * R2));

        // Pseudo-inverse via eigendecomposition, stored in the N+1'th slot of
        // the 2D M array.

        LAPACKE_dsyev( CblasRowMajor, 
                        'V', 
                        'U', 
                        R,
                        M(), 
                        R, 
                        lambda() );

        #pragma omp parallel for
        for(uint32_t v = 0; v < R; v++) {
            if(lambda[v] > eigenvalue_tolerance) {
                for(uint32_t u = 0; u < R; u++) {
                        M[N * R2 + u * R + R - 1 - v] = M[u * R + v] / sqrt(lambda[v]); 
                }
            }
            else {
                for(uint32_t u = 0; u < R; u++) {
                        M[N * R2 + u * R + R - 1 - v] = 0.0; 
                }
            }
        }

        cblas_dsyrk(CblasRowMajor, 
                    CblasUpper, 
                    CblasNoTrans,
                    R,
                    R, 
                    1.0, 
                    (const double*) M(N, 0), 
                    R, 
                    0.0, 
                    M(), 
                    R);

        #pragma omp parallel
{
        for(uint32_t k = N - 1; k > 0; k--) {
            if(k != j) {
                #pragma omp for
                for(uint32_t i = 0; i < R2; i++) {
                    M[k * R2 + i] *= M[i];   
                }
            }
        }

        // Eigendecompose each of the gram matrices 
        #pragma omp for
        for(uint32_t k = N; k > 0; k--) {
            if(k != j) {
                if(k < N) {
                    LAPACKE_dsyev( CblasRowMajor, 
                                    'V', 
                                    'U', 
                                    R,
                                    M(k, 0), 
                                    R, 
                                    lambda(k, 0) );

                    for(uint32_t v = 0; v < R; v++) { 
                        for(uint32_t u = 0; u < R; u++) {
                            M[k * R2 + u * R + v] *= sqrt(lambda[k * R + v]); 
                        }
                    }
                }
                transpose_square_in_place(M(k, 0), R);
            }
        }
}

        for(int k = N-1; k >= 0; k--) {
            if((uint32_t) k != j) {
                int offset = (k + 1 == (int) j) ? k + 2 : k + 1;
                eigen_trees[k]->build_tree(scaled_eigenvecs[offset]);
                eigen_trees[k]->multiply_matrices_against_provided(gram_trees[k]->G);
            }
        } 
    }

    void fill_buffer_random_draws(double* data, uint64_t len) {
        for(uint64_t i = 0; i < len; i++) {
            data[i] = dis(gen);
        }
    }

    void KRPDrawSamples(uint32_t j, Buffer<uint64_t> &samples, Buffer<double> *random_draws) {
        // Samples is an array of size N x J 
        computeM(j);
        std::fill(h(), h(J, 0), 1.0);

        for(uint32_t k = 0; k < N; k++) {
            if(k != j) {
                // Sample an eigenvector component of the mixture distribution 
                std::copy(h(), h(J, 0), scaled_h());
                Buffer<uint64_t> row_buffer({J}, samples(k, 0));
                int offset = (k + 1 == j) ? k + 2 : k + 1;

                if(random_draws != nullptr) {
                    Buffer<double> eigen_draws({J}, (*random_draws)(k * J));    
                    eigen_trees[k]->PTSample_internal(scaled_eigenvecs[offset], 
                            scaled_h,
                            h,
                            row_buffer,
                            eigen_draws
                            );
                    Buffer<double> gram_draws({J}, (*random_draws)(N * J + k * J));
                    gram_trees[k]->PTSample_internal(U[k], 
                            h,
                            scaled_h,
                            row_buffer,
                            gram_draws
                            );
                }
                else {
                    fill_buffer_random_draws(scratch.random_draws(), J);
                    eigen_trees[k]->PTSample_internal(scaled_eigenvecs[offset], 
                            scaled_h,
                            h,
                            row_buffer,
                            scratch.random_draws 
                            );
                    fill_buffer_random_draws(scratch.random_draws(), J);
                    gram_trees[k]->PTSample_internal(U[k], 
                            h,
                            scaled_h,
                            row_buffer,
                            scratch.random_draws 
                            );
                }
            }
        }

        // Compute the weights associated with the samples
        compute_DAGAT(
            h(),
            M(),
            weights(),
            J,
            R);

        #pragma omp parallel for 
        for(uint32_t i = 0; i < J; i++) {
            weights[i] = (double) R / (weights[i] * J);
        }
    }

    ~EfficientKRPSampler() {
        for(uint32_t i = 0; i < N; i++) {
            delete gram_trees[i];
            delete eigen_trees[i];
        }
    }
};