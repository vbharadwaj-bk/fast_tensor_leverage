#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <initializer_list>
#include <chrono>
#include <random>
#include <memory>
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

inline uint32_t divide_and_roundup(uint32_t n, uint32_t m) {
    return (n + m - 1) / m;
}

inline void log2_round_down(uint32_t m, 
        uint32_t& log2_res, 
        uint32_t& lowest_power_2) {
    
    assert(m > 0);
    log2_res = 0;
    lowest_power_2 = 1;

    while(lowest_power_2 * 2 <= m) {
        log2_res++; 
        lowest_power_2 *= 2;
    }
}

//#pragma GCC visibility push(hidden)
template<typename T>
class __attribute__((visibility("hidden"))) Buffer {
    unique_ptr<T[]> managed_ptr;
    T* ptr;
    uint64_t dim0;
    uint64_t dim1;

public:
    vector<uint64_t> shape;

    Buffer(Buffer&& other)
        :   managed_ptr(std::move(other.managed_ptr)),
            ptr(std::move(other.ptr)),
            dim0(other.dim0),
            dim1(other.dim1),
            shape(std::move(other.shape))
    {}
    Buffer& operator=(const Buffer& other) = default;

    Buffer(initializer_list<uint64_t> args) {
        uint64_t buffer_size = 1;
        for(uint64_t i : args) {
            buffer_size *= i;
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        managed_ptr.reset(new T[buffer_size]);
        ptr = managed_ptr.get();
    }

    Buffer(initializer_list<uint64_t> args, T* ptr) {
        for(uint64_t i : args) {
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        this->ptr = ptr;
    }

    T* operator()() {
        return ptr;
    }

    T* operator()(uint64_t offset) {
        return ptr + offset;
    }

    // Assumes that this array is a row-major matrix 
    T* operator()(uint64_t off_x, uint64_t off_y) {
        return ptr + (dim1 * off_x) + off_y;
    }

    T& operator[](uint64_t offset) {
        return ptr[offset];
    }

    ~Buffer() {
    }
};

void compute_DAGAT(double* A, double* G, 
        double* res, uint64_t J, uint64_t R) {

    Buffer<double> temp({J, R});

    cblas_dsymm(
        CblasRowMajor,
        CblasRight,
        CblasUpper,
        (uint32_t) J,
        (uint32_t) R,
        1.0,
        G,
        R,
        A,
        R,
        0.0,
        temp(),
        R
    );

    #pragma omp parallel for 
    for(uint32_t i = 0; i < J; i++) {
        res[i] = 0.0;
        for(uint32_t j = 0; j < R; j++) {
            res[i] += A[i * R + j] * temp[i * R + j];
        }
    }
}



/*
* exclude is the index of a matrix to exclude from the chain Hadamard product. Pass -1
* to include all components in the chain Hadamard product.
*/
void ATB_chain_prod(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B,
        Buffer<double> &result,
        int exclude) {

        uint64_t N = A.size();
        uint64_t R_A = A[0].shape[1];
        uint64_t R_B = B[0].shape[1];

        vector<unique_ptr<Buffer<double>>> ATB;
        for(uint64_t i = 0; i < A.size(); i++) {
                ATB.emplace_back();
                ATB[i].reset(new Buffer<double>({R_A, R_B}));
        }

        for(uint64_t i = 0; i < R_A; i++) {
                for(uint64_t j = 0; j < R_B; j++) {
                        result[i * R_B + j] = sigma_A[i] * sigma_B[j];
                }

        }

        // Can replace with a batch DGEMM call
        for(uint64_t i = 0; i < N; i++) {
            if(((int) i) != exclude) {
                uint64_t K = A[i].shape[0];
                cblas_dgemm(
                        CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        R_A,
                        R_B,
                        K,
                        1.0,
                        A[i](),
                        R_A,
                        B[i](),
                        R_B,
                        0.0,
                        (*(ATB[i]))(),
                        R_B
                );
            }
        }

        #pragma omp parallel 
{
        for(uint64_t k = 0; k < N; k++) {
                if(((int) k) != exclude) {
                    #pragma omp for collapse(2)
                    for(uint64_t i = 0; i < R_A; i++) {
                            for(uint64_t j = 0; j < R_B; j++) {
                                    result[i * R_B + j] *= (*(ATB[k]))[i * R_B + j];
                            }
                    }
                }
        }
}
}

double ATB_chain_prod_sum(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B) {

    uint64_t R_A = A[0].shape[1];
    uint64_t R_B = B[0].shape[1];
    Buffer<double> result({R_A, R_B});
    ATB_chain_prod(A, B, sigma_A, sigma_B, result, -1);
    return std::accumulate(result(), result(R_A * R_B), 0.0); 
}

void compute_pinv_square(Buffer<double> &M, Buffer<double> &out, uint64_t target_rank) {
    uint64_t R = M.shape[0];
    double eigenvalue_tolerance = 1e-10;
    Buffer<double> lambda({R});

    LAPACKE_dsyev( CblasRowMajor, 
                    'V', 
                    'U', 
                    R,
                    M(), 
                    R, 
                    lambda() );

    //cout << "Lambda: ";
    for(uint32_t v = 0; v < R; v++) {
        //cout << lambda[v] << " ";
        if(v >= R - target_rank && lambda[v] > eigenvalue_tolerance) {
            for(uint32_t u = 0; u < R; u++) {
                M[u * R + v] = M[u * R + v] / sqrt(lambda[v]); 
            }
        }
        else {
            for(uint32_t u = 0; u < R; u++) {
                M[u * R + v] = 0.0; 
            }
        }
    }
    //cout << "]" << endl;

    cblas_dsyrk(CblasRowMajor, 
                CblasUpper, 
                CblasNoTrans,
                R,
                R, 
                1.0, 
                (const double*) M(), 
                R, 
                0.0, 
                out(), 
                R);

}

void compute_pinv(Buffer<double> &in, Buffer<double> &out) {
    uint64_t R = in.shape[1];
    Buffer<double> M({R, R});

    std::fill(M(), M(R * R), 0.0);

    uint64_t I = in.shape[0];
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        uint64_t work = (I + num_threads - 1) / num_threads;
        uint64_t start = min(work * thread_id, I);
        uint64_t end = min(work * (thread_id + 1), I);

        if(end - start > 0) {
            Buffer<double> local({R, R});
            cblas_dsyrk(CblasRowMajor, 
                        CblasUpper, 
                        CblasTrans,
                        R,
                        end-start, 
                        1.0, 
                        in(start * R), 
                        R, 
                        0.0, 
                        local(), 
                        R);

            for(uint64_t i = 0; i < R * R; i++) {
                #pragma omp atomic
                M[i] += local[i];
            }
        }

    }

    uint64_t target_rank = min(in.shape[0], R);
    compute_pinv_square(M, out, target_rank);
}

class __attribute__((visibility("hidden"))) ScratchBuffer {
public:
    Buffer<int64_t> c;
    Buffer<double> temp1;
    Buffer<double> q;
    Buffer<double> m;
    Buffer<double> mL;
    Buffer<double> low;
    Buffer<double> high;
    Buffer<double> random_draws;
    Buffer<double*> a_array;
    Buffer<double*> x_array;
    Buffer<double*> y_array;

    ScratchBuffer(uint32_t F, uint64_t J, uint64_t R) : 
            c({J}),
            temp1({J, R}),
            q({J, F}),
            m({J}),
            mL({J}),
            low({J}),
            high({J}),
            random_draws({J}),
            a_array({J}),
            x_array({J}),
            y_array({J}) 
    {}
};

class __attribute__((visibility("hidden"))) PartitionTree {
public:
    int64_t n, F;
    uint32_t leaf_count, node_count;

    uint32_t lfill_level, lfill_count;
    uint32_t total_levels;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;

    int64_t J, R;
    int64_t R2;

    Buffer<double> G;
    ScratchBuffer &scratch;

    unique_ptr<Buffer<double>> G_unmultiplied;

    void execute_mkl_dsymv_batch() {
        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            cblas_dsymv(CblasRowMajor, 
                    CblasUpper, 
                    R, 
                    1.0, 
                    (const double*) scratch.a_array[i],
                    R, 
                    (const double*) scratch.x_array[i], 
                    1, 
                    0.0, 
                    scratch.y_array[i], 
                    1);
        }
    }

    void execute_mkl_dgemv_batch() {
        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            cblas_dgemv(CblasRowMajor, 
                    CblasNoTrans,
                    F,
                    R, 
                    1.0, 
                    (const double*) scratch.a_array[i],
                    R, 
                    (const double*) scratch.x_array[i], 
                    1, 
                    0.0, 
                    scratch.y_array[i], 
                    1);
        }
    }

    PartitionTree(uint32_t n, uint32_t F, uint64_t J, uint64_t R, ScratchBuffer &scr)
        :   G({2 * divide_and_roundup(n, F) - 1, R * R}),
            scratch(scr)
        {
        this->n = n;
        this->F = F;
        this->J = J;
        this->R = R;
        R2 = R * R;

        leaf_count = divide_and_roundup(n, F);
        node_count = 2 * leaf_count - 1;

        log2_round_down(leaf_count, lfill_level, lfill_count);
        total_levels = node_count > lfill_count ? lfill_level + 1 : lfill_level;

        nodes_upto_lfill = lfill_count * 2 - 1;
        nodes_before_lfill = lfill_count - 1;

        uint32_t nodes_at_partial_level_div2 = (node_count - nodes_upto_lfill) / 2;
        complete_level_offset = nodes_before_lfill - nodes_at_partial_level_div2;
        G_unmultiplied.reset(nullptr);
    }

    bool is_leaf(int64_t c) {
        return 2 * c + 1 >= node_count; 
    }

    int64_t leaf_idx(int64_t c) {
        if(c >= nodes_upto_lfill) {
            return c - nodes_upto_lfill;
        }
        else {
            return c - complete_level_offset; 
        }
    }

    void build_tree(Buffer<double> &U) {
        G_unmultiplied.reset(nullptr);

        // First leaf must always be on the lowest filled level 
        int64_t first_leaf_idx = node_count - leaf_count; 

        Buffer<double*> a_array({leaf_count});
        Buffer<double*> c_array({leaf_count});

        #pragma omp parallel
{
        #pragma omp for
        for(int64_t i = 0; i < node_count * R2; i++) {
            G[i] = 0.0;
        }

        #pragma omp for
        for(int64_t i = 0; i < leaf_count; i++) {
            uint64_t idx = leaf_idx(first_leaf_idx + i);
            a_array[i] = U(idx * F, 0);
            c_array[i] = G(first_leaf_idx + i, 0);
        }
 
        #pragma omp for
        for(int64_t i = 0; i < leaf_count; i++) {
            cblas_dsyrk(CblasRowMajor, 
                        CblasUpper, 
                        CblasTrans,
                        R,
                        F, 
                        1.0, 
                        (const double*) a_array[i], 
                        R, 
                        0.0, 
                        c_array[i], 
                        R);
        }

        int64_t start = nodes_before_lfill; 
        int64_t end = first_leaf_idx;

        for(int c_level = lfill_level; c_level >= 0; c_level--) {
            #pragma omp for 
            for(int c = start; c < end; c++) {
                for(int j = 0; j < R2; j++) {
                    G[c * R2 + j] += G[(2 * c + 1) * R2 + j] + G[(2 * c + 2) * R2 + j];
                } 
            }
            end = start;
            start = ((start + 1) / 2) - 1;
        }
}
    }

    /*
    * Multiplies the partial gram matrices maintained by this tree against
    * the provided buffer, caching the old values for future multiplications. 
    */
    void multiply_matrices_against_provided(Buffer<double> &mat) {
        if(! G_unmultiplied) {
            G_unmultiplied.reset(new Buffer<double>({node_count, static_cast<unsigned long>(R2)}));
            std::copy(G(), G(node_count * R2), (*G_unmultiplied)());
        }
        #pragma omp parallel for
        for(int64_t i = 0; i < node_count; i++) {
            for(int j = 0; j < R2; j++) {
                G[i * R2 + j] = (*G_unmultiplied)[i * R2 + j] * mat[j];
            }
        }
    }

    void batch_dot_product(
                double* A, 
                double* B, 
                double* result,
                int64_t J, int64_t R 
                ) {
        #pragma omp for
        for(int i = 0; i < J; i++) {
            result[i] = 0;
            for(int j = 0; j < R; j++) {
                result[i] += A[i * R + j] * B[i * R + j];
            }
        }
    }

    void PTSample_internal(Buffer<double> &U, 
            Buffer<double> &h,
            Buffer<double> &scaled_h,
            Buffer<uint64_t> &samples,
            Buffer<double> &random_draws
            ) {
 
        Buffer<int64_t> &c = scratch.c;
        Buffer<double> &temp1 = scratch.temp1;
        Buffer<double> &q = scratch.q;
        Buffer<double> &m = scratch.m;
        Buffer<double> &mL = scratch.mL;
        Buffer<double> &low = scratch.low;
        Buffer<double> &high = scratch.high;
        //Buffer<double> &random_draws = scratch.random_draws;
        Buffer<double*> &a_array = scratch.a_array;
        Buffer<double*> &x_array = scratch.x_array;
        Buffer<double*> &y_array = scratch.y_array;

        #pragma omp parallel
{
        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            x_array[i] = scaled_h(i, 0);
            y_array[i] = temp1(i, 0); 

            c[i] = 0;
            low[i] = 0.0;
            high[i] = 1.0;
            a_array[i] = G(0);
        }

        execute_mkl_dsymv_batch();

        batch_dot_product(
            scaled_h(), 
            temp1(), 
            m(),
            J, R 
            );

        for(uint32_t c_level = 0; c_level < lfill_level; c_level++) {
            // Prepare to compute m(L(v)) for all v

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                a_array[i] = G((2 * c[i] + 1) * R2); 
            }

            execute_mkl_dsymv_batch();

            batch_dot_product(
                scaled_h(), 
                temp1(), 
                mL(),
                J, R 
                );

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                double cutoff = low[i] + mL[i] / m[i];
                if(random_draws[i] <= cutoff) {
                    c[i] = 2 * c[i] + 1;
                    high[i] = cutoff;
                }
                else {
                    c[i] = 2 * c[i] + 2;
                    low[i] = cutoff;
                }
            }
        }

        // Handle the tail case
        if(node_count > nodes_before_lfill) {
            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                a_array[i] = is_leaf(c[i]) ? a_array[i] : G((2 * c[i] + 1) * R2); 
            }

            execute_mkl_dsymv_batch();

            batch_dot_product(
                scaled_h(), 
                temp1(), 
                mL(),
                J, R 
                );

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                double cutoff = low[i] + mL[i] / m[i];
                if((! is_leaf(c[i])) && random_draws[i] <= cutoff) {
                    c[i] = 2 * c[i] + 1;
                    high[i] = cutoff;
                }
                else if((! is_leaf(c[i])) && random_draws[i] > cutoff) {
                    c[i] = 2 * c[i] + 2;
                    low[i] = cutoff;
                }
            }
        }

        // We will use the m array as a buffer 
        // for the draw fractions.
        if(F > 1) {
            #pragma omp for
            for(int i = 0; i < J; i++) {
                m[i] = (random_draws[i] - low[i]) / (high[i] - low[i]);

                int64_t leaf_idx;
                if(c[i] >= nodes_upto_lfill) {
                    leaf_idx = c[i] - nodes_upto_lfill;
                }
                else {
                    leaf_idx = c[i] - complete_level_offset; 
                }

                a_array[i] = U(leaf_idx * F, 0);
                y_array[i] = q(i * F);
            }

            execute_mkl_dgemv_batch();
        }
        
        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            int64_t res; 
            if(F > 1) {
                res = F - 1;
                double running_sum = 0.0;
                for(int64_t j = 0; j < F; j++) {
                    double temp = q[i * F + j] * q[i * F + j];
                    q[i * F + j] = running_sum;
                    running_sum += temp;
                }

                for(int64_t j = 0; j < F; j++) {
                    q[i * F + j] /= running_sum; 
                }

                for(int64_t j = 0; j < F - 1; j++) {
                    if(m[i] < q[i * F + j + 1]) {
                        res = j; 
                        break;
                    }
                }
            }
            else {
                res = 0;
            }

            int64_t idx = leaf_idx(c[i]);
            samples[i] = res + idx * F;
            
            for(int64_t j = 0; j < R; j++) {
                h[i * R + j] *= U[(res + idx * F) * R + j];
            }  
        }
}
    }
};

class __attribute__((visibility("hidden"))) Sampler {
public:
    uint64_t N, J, R, R2;
    vector<Buffer<double>> &U;
    Buffer<double> h;
    Buffer<double> weights;

    // Related to random number generation 
    std::random_device rd;  
    std::mt19937 gen;

    // Related to independent random number generation on multiple
    // streams
    int thread_count;
    vector<std::mt19937> par_gen; 

    Sampler(uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices) : 
        U(U_matrices),
        h({J, R}),
        weights({J}),
        rd(),
        gen(rd())
        {
        this->N = U.size();
        this->J = J;
        this->R = R;
        R2 = R * R;


        // Set up independent random streams for different threads.
        // As written, might be more complicated than it needs to be. 
        #pragma omp parallel
        {
            #pragma omp single 
            {
                thread_count = omp_get_num_threads();
            }
        }

        vector<uint32_t> biased_seeds(thread_count, 0);
        vector<uint32_t> seeds(thread_count, 0);

        for(int i = 0; i < thread_count; i++) {
            biased_seeds[i] = rd();
        }
        std::seed_seq seq(biased_seeds.begin(), biased_seeds.end());
        seq.generate(seeds.begin(), seeds.end());

        for(int i = 0; i < thread_count; i++) {
            par_gen.emplace_back(seeds[i]);
        }
    }

    virtual void update_sampler(uint64_t j) = 0;
    virtual void KRPDrawSamples(uint32_t j, Buffer<uint64_t> &samples, Buffer<double> *random_draws) = 0;

    /*
    * Fills the h matrix based on an array of samples. Can be bypassed if KRPDrawSamples computes
    * h during its execution.
    */
    void fill_h_by_samples(Buffer<uint64_t> &samples, uint64_t j) {
        std::fill(h(), h(J * R), 1.0);
        for(uint32_t k = 0; k < N; k++) {
            if(k != j) {
                Buffer<uint64_t> row_buffer({J}, samples(k, 0)); // View into a row of the samples array

                #pragma omp parallel for 
                for(uint64_t i = 0; i < J; i++) {
                    uint64_t sample = row_buffer[i];
                    for(uint64_t u = 0; u < R; u++) {
                        h[i * R + u] *= U[k][sample * R + u];
                    }
                }
            }
        }
    } 
};


class __attribute__((visibility("hidden"))) EfficientKRPSampler : public Sampler {
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
        #pragma omp parallel
{
        int thread_id = omp_get_thread_num();
        auto &local_gen = par_gen[thread_id];

        #pragma omp for
        for(uint64_t i = 0; i < len; i++) {
            data[i] = dis(local_gen);
        }
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

int main(int argc, char** argv) {
    vector<Buffer<double>> U;

    uint64_t I = 50000000;
    uint64_t J = 10000;
    uint64_t R = 25;
    uint64_t N = 4;

    Buffer<uint64_t> samples({J, N});

    for(uint64_t i = 0; i < N; i++) {
      U.emplace_back(std::initializer_list<uint64_t>({I, R}));
    }

    unique_ptr<EfficientKRPSampler> sampler;
    sampler.reset(new EfficientKRPSampler(J, R, U)); 
    sampler->KRPDrawSamples(0, samples, nullptr); 
    sampler.reset(nullptr);
    cout << "Built sampler!" << endl;
}