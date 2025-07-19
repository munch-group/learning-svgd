// ptdalgorithms_jax.cc
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "ptdalgorithms_jax.h"

extern "C" {

// JAX custom call signature for ptdalgorithms_jax
void ptdalgorithms_jax(void* out_ptr, void** in_ptrs) {
    // Input 0: theta array (shape varies, contains alpha, T, t concatenated)
    // Input 1: times array (shape: [n])
    // Input 2: m scalar (int64)
    // Input 3: n scalar (int64)

    double* theta = reinterpret_cast<double*>(in_ptrs[0]);
    int64_t* times = reinterpret_cast<int64_t*>(in_ptrs[1]);
    int64_t* m_ptr = reinterpret_cast<int64_t*>(in_ptrs[2]);
    int64_t* n_ptr = reinterpret_cast<int64_t*>(in_ptrs[3]);
    double* output = reinterpret_cast<double*>(out_ptr);
    
    // Extract dimensions from scalar operands
    int64_t m = *m_ptr;
    int64_t n = *n_ptr;
    
    //////////
    const std::string filename = "models.h5";
    std::vector<double> params(theta, theta + m);
    const std::string key = hash_key_from_input(params);    
    if (key_exists(filename, key)) {
        std::cout << "Key exists: ";
        std::cout << key;
        std::cout << std::endl;
        MyModel loaded_model = HDF5ModelStore::load(filename, key);

        std::cout << "Loaded model: ID=" << loaded_model.id;
        //         << ", Name=" << loaded_model.name
        //         << ", Weights=[";
        // for (const auto& w : loaded_model.weights) std::cout << w << " ";
        // std::cout << "]\n";        
        std::cout << std::endl;
    } else {
        std::cout << "Key does not exist\n";

        MyModel model_to_save = {42, "example", {0.1, 0.2, 0.3}};
        HDF5ModelStore::save(filename, key, model_to_save);
    }
    //////////


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
void register_ptdalgorithms_jax() {
    // This would normally register with XLA, but for simplicity we'll rely on 
    // the Python side custom call mechanism
}

}




