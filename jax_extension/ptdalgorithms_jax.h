#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

// JAX custom call signature with scalar operands
__attribute__((visibility("default")))
void ptdalgorithms_jax(void* out_ptr, void** in_ptrs);

#ifdef __cplusplus
}
#endif

/////////////
// hdf5_model_store.hpp
#pragma once
#include <H5Cpp.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>
#include <mutex>
struct MyModel {
    int id;
    std::string name;
    std::vector<double> weights;
};

// std::string hash_key_from_input(const std::vector<double>& inputs) {
//     std::ostringstream oss;
//     for (double x : inputs) {
//         oss << std::setprecision(17) << x << ",";
//     }

//     std::string str = oss.str();
//     unsigned char hash[SHA256_DIGEST_LENGTH];
//     SHA256((const unsigned char*)str.data(), str.size(), hash);

//     std::ostringstream key;
//     key << "key_";
//     for (int i = 0; i < 8; ++i) {  // short 64-bit prefix
//         key << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
//     }
//     return key.str();
// }

std::string hash_key_from_input(const std::vector<double>& vec) {
    std::ostringstream oss;
    oss << std::setprecision(4) << std::scientific;  // full double precision
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) oss << "_";
        oss << vec[i];
    }

    return oss.str();
}

bool key_exists(const std::string& filename, const std::string& key) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        return file.nameExists(key);
    } catch (...) {
        return false;
    }
}

class HDF5ModelStore {
public:
    static void cache_model(const std::string& key, const MyModel& model) {
        std::lock_guard<std::mutex> lock(mem_mutex);
        mem_cache[key] = model;
    }

    static bool in_memory(const std::string& key) {
        std::lock_guard<std::mutex> lock(mem_mutex);
        return mem_cache.find(key) != mem_cache.end();
    }

    static MyModel get_cached(const std::string& key) {
        std::lock_guard<std::mutex> lock(mem_mutex);
        return mem_cache.at(key);
    }

    static MyModel load_cached(const std::string& filename, const std::string& key) {
        {
            std::lock_guard<std::mutex> lock(mem_mutex);
            if (mem_cache.find(key) != mem_cache.end()) {
                return mem_cache.at(key);
            }
        }

        MyModel model = load(filename, key);

        {
            std::lock_guard<std::mutex> lock(mem_mutex);
            mem_cache[key] = model;
        }

        return model;
    }

    static void save(const std::string& filename, const std::string& key, const MyModel& model) {
        std::lock_guard<std::mutex> lock(mem_mutex);
        H5::H5File file;
        try {
            file = H5::H5File(filename, H5F_ACC_RDWR);
        } catch (...) {
            file = H5::H5File(filename, H5F_ACC_TRUNC);
        }

        if (!file.nameExists(key)) {
            file.createGroup(key);
        }

        hsize_t dims[1] = {model.weights.size()};
        H5::DataSpace wspace(1, dims);
        H5::DataSet wset = file.createDataSet(key + "/weights", H5::PredType::NATIVE_DOUBLE, wspace);
        wset.write(model.weights.data(), H5::PredType::NATIVE_DOUBLE);

        hsize_t id_dims[1] = {1};
        H5::DataSpace id_space(1, id_dims);
        H5::DataSet idset = file.createDataSet(key + "/id", H5::PredType::NATIVE_INT, id_space);
        idset.write(&model.id, H5::PredType::NATIVE_INT);

        hsize_t str_dims[1] = {model.name.size()};
        H5::StrType str_type(H5::PredType::C_S1, model.name.size());
        H5::DataSpace str_space(1, str_dims);
        H5::DataSet strset = file.createDataSet(key + "/name", str_type, str_space);
        strset.write(model.name, str_type);
    }

    static MyModel load(const std::string& filename, const std::string& key) {
        std::lock_guard<std::mutex> lock(mem_mutex);
        H5::H5File file(filename, H5F_ACC_RDONLY);

        H5::DataSet idset = file.openDataSet(key + "/id");
        int id;
        idset.read(&id, H5::PredType::NATIVE_INT);

        H5::DataSet strset = file.openDataSet(key + "/name");
        H5::StrType str_type = strset.getStrType();
        std::string name;
        strset.read(name, str_type);

        H5::DataSet wset = file.openDataSet(key + "/weights");
        H5::DataSpace wspace = wset.getSpace();
        hsize_t dims[1];
        wspace.getSimpleExtentDims(dims);
        std::vector<double> weights(dims[0]);
        wset.read(weights.data(), H5::PredType::NATIVE_DOUBLE);

        return {id, name, weights};
    }

    static bool key_exists(const std::string& filename, const std::string& key) {
        std::lock_guard<std::mutex> lock(mem_mutex);
        try {
            H5::H5File file(filename, H5F_ACC_RDONLY);
            return file.nameExists(key);
        } catch (...) {
            return false;
        }
    }
private:
    static std::unordered_map<std::string, MyModel> mem_cache;
    static std::mutex mem_mutex;

};

std::unordered_map<std::string, MyModel> HDF5ModelStore::mem_cache;
std::mutex HDF5ModelStore::mem_mutex;

// class HDF5ModelStore {
//     public:
//         static void save(const std::string& filename, const std::string& key, const MyModel& model) {
//             std::lock_guard<std::mutex> lock(hdf5_mutex);
//             H5::H5File file;
//             try {
//                 file = H5::H5File(filename, H5F_ACC_RDWR);
//             } catch (...) {
//                 file = H5::H5File(filename, H5F_ACC_TRUNC);
//             }

//             if (!file.nameExists(key)) {
//                 file.createGroup(key);
//             }

//             hsize_t dims[1] = {model.weights.size()};
//             H5::DataSpace wspace(1, dims);
//             H5::DataSet wset = file.createDataSet(key + "/weights", H5::PredType::NATIVE_DOUBLE, wspace);
//             wset.write(model.weights.data(), H5::PredType::NATIVE_DOUBLE);

//             hsize_t id_dims[1] = {1};
//             H5::DataSpace id_space(1, id_dims);
//             H5::DataSet idset = file.createDataSet(key + "/id", H5::PredType::NATIVE_INT, id_space);
//             idset.write(&model.id, H5::PredType::NATIVE_INT);

//             hsize_t str_dims[1] = {model.name.size()};
//             H5::StrType str_type(H5::PredType::C_S1, model.name.size());
//             H5::DataSpace str_space(1, str_dims);
//             H5::DataSet strset = file.createDataSet(key + "/name", str_type, str_space);
//             strset.write(model.name, str_type);
//         }

//         static MyModel load(const std::string& filename, const std::string& key) {
//             std::lock_guard<std::mutex> lock(hdf5_mutex);
//             H5::H5File file(filename, H5F_ACC_RDONLY);

//             H5::DataSet idset = file.openDataSet(key + "/id");
//             int id;
//             idset.read(&id, H5::PredType::NATIVE_INT);

//             H5::DataSet strset = file.openDataSet(key + "/name");
//             H5::StrType str_type = strset.getStrType();
//             std::string name;
//             strset.read(name, str_type);

//             H5::DataSet wset = file.openDataSet(key + "/weights");
//             H5::DataSpace wspace = wset.getSpace();
//             hsize_t dims[1];
//             wspace.getSimpleExtentDims(dims);
//             std::vector<double> weights(dims[0]);
//             wset.read(weights.data(), H5::PredType::NATIVE_DOUBLE);

//             return {id, name, weights};
//         }

//         static bool key_exists(const std::string& filename, const std::string& key) {
//             std::lock_guard<std::mutex> lock(hdf5_mutex);
//             try {
//                 H5::H5File file(filename, H5F_ACC_RDONLY);
//                 return file.nameExists(key);
//             } catch (...) {
//                 return false;
//             }
//         }

//     private:
//         static std::mutex hdf5_mutex;
// };

// std::mutex HDF5ModelStore::hdf5_mutex;
