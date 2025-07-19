// dph_pmf_param.cc
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <cassert>

extern "C" {

// JAX custom call signature for dph_pmf_param
void dph_pmf_param(void* out_ptr, void** in_ptrs, const char* opaque, size_t opaque_len) {
    // Input 0: theta array (shape varies, contains alpha, T, t concatenated)
    // Input 1: times array (shape: [n])
    
    double* theta = reinterpret_cast<double*>(in_ptrs[0]);
    int64_t* times = reinterpret_cast<int64_t*>(in_ptrs[1]);
    double* output = reinterpret_cast<double*>(out_ptr);
    
    // Extract m and n from opaque data
    assert(opaque_len == 2 * sizeof(int64_t));
    int64_t m, n;
    std::memcpy(&m, opaque, sizeof(int64_t));
    std::memcpy(&n, opaque + sizeof(int64_t), sizeof(int64_t));
    
    std::vector<double> a(m), temp(m);

    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t k = times[idx];
        
        // Copy alpha (first m elements of theta)
        std::copy(theta, theta + m, a.begin());

        // Matrix multiplication k times: a = a * T^k
        for (int64_t step = 0; step < k; ++step) {
            std::fill(temp.begin(), temp.end(), 0.0);
            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < m; ++j) {
                    temp[j] += a[i] * theta[m + i * m + j]; // T matrix starts at theta[m]
                }
            }
            std::swap(a, temp);
        }

        // Final probability: a * t (exit probabilities start at theta[m + m*m])
        double pmf = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            pmf += a[i] * theta[m + m * m + i]; // t vector starts at theta[m + m*m]
        }
        output[idx] = pmf;
    }


}

// XLA custom call registration
void register_dph_pmf_param() {
    // This would normally register with XLA, but for simplicity we'll rely on 
    // the Python side custom call mechanism
}

}



// extern "C" {

// // Computes PMF of a discrete phase-type distribution for integer times
// // alpha: initial distribution vector (length m)
// // T: sub-transition matrix (m x m), rows sum to <=1
// // t: exit probabilities (length m), with T[i,*] + t[i] = 1
// // times: integer array (length N), values t_1,...,t_N
// // output: same shape as times, PMF(t_i) for each t_i
// void dph_pmf_param(const double* alpha, const double* T, const double* t,
//                    const int64_t* times, double* output,
//                    int64_t m, int64_t n) {
//     std::vector<double> a(m), b(m), temp(m);

//     for (int64_t idx = 0; idx < n; ++idx) {
//         int64_t k = times[idx];
//         std::copy(alpha, alpha + m, a.begin());

//         for (int64_t step = 0; step < k; ++step) {
//             std::fill(temp.begin(), temp.end(), 0.0);
//             for (int64_t i = 0; i < m; ++i) {
//                 for (int64_t j = 0; j < m; ++j) {
//                     temp[j] += a[i] * T[i * m + j];
//                 }
//             }
//             std::swap(a, temp);
//         }

//         double pmf = 0.0;
//         for (int64_t i = 0; i < m; ++i) {
//             pmf += a[i] * t[i];
//         }
//         output[idx] = pmf;
//     }
// }

// }

// #include <cmath>
// #include <cstdint>
// #include <mutex>
// #include <vector>
// #include <cassert>
// #include <cstring>
// #include "ptdalgorithms/ptdalgorithms.h"

// extern "C" void dph_pmf_param(
//     void* out_ptr,
//     void** in_ptrs,
//     int64_t* shape_ptr,
//     int ndims
// ) {
//     assert(ndims == 1);
//     int64_t N = shape_ptr[0];

//     // Inputs
//     double* alpha_in = reinterpret_cast<double*>(in_ptrs[0]); // shape: (m,)
//     double* T_in     = reinterpret_cast<double*>(in_ptrs[1]); // shape: (m * m,)
//     double* t_in     = reinterpret_cast<double*>(in_ptrs[2]); // shape: (m,)
//     int64_t* times   = reinterpret_cast<int64_t*>(in_ptrs[3]); // shape: (N,)

//     double* out      = reinterpret_cast<double*>(out_ptr);

//     const int m = static_cast<int>(shape_ptr[1]);  // m passed via shape_ptr[1]

//     // Build the DPH model graph from α, T, t
//     ptdalgorithms::Graph g;
//     std::vector<int> transient_states(m);
//     for (int i = 0; i < m; ++i) transient_states[i] = i;

//     int absorbing = m;
//     g.AddStates(transient_states);
//     g.AddAbsorbingStates({absorbing});

//     for (int i = 0; i < m; ++i) {
//         double total = 0.0;
//         for (int j = 0; j < m; ++j) {
//             double p = T_in[i * m + j];
//             if (p > 0.0) g.AddTransition(i, j, p);
//             total += p;
//         }
//         double to_absorb = t_in[i];
//         if (to_absorb > 0.0 || 1.0 - total > 1e-12) {
//             g.AddTransition(i, absorbing, to_absorb);
//         }
//     }

//     // Set initial distribution
//     g.SetStartingDistribution(std::vector<double>(alpha_in, alpha_in + m));

//     // Precompute PMF
//     auto pmf = g.GetDiscretePhaseTypePMF(N + 100); // Overshoot allowed

//     for (int64_t i = 0; i < N; ++i) {
//         int t = static_cast<int>(times[i]);
//         out[i] = t < static_cast<int>(pmf.size()) ? pmf[t] : 0.0;
//     }
// }

