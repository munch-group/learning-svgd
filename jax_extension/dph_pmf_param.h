#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

// JAX custom call signature with scalar operands
__attribute__((visibility("default")))
void dph_pmf_param(void* out_ptr, void** in_ptrs);

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
struct MyModel {
    int id;
    std::string name;
    std::vector<double> weights;
};

std::string hash_key_from_input(const std::vector<double>& inputs) {
    std::ostringstream oss;
    for (double x : inputs) {
        oss << std::setprecision(17) << x << ",";
    }

    std::string str = oss.str();
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256((const unsigned char*)str.data(), str.size(), hash);

    std::ostringstream key;
    key << "key_";
    for (int i = 0; i < 8; ++i) {  // short 64-bit prefix
        key << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return key.str();
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
    static void save(const std::string& filename, const std::string& key, const MyModel& model) {
        H5::H5File file;
        try {
            file = H5::H5File(filename, H5F_ACC_RDWR);
        } catch (...) {
            file = H5::H5File(filename, H5F_ACC_TRUNC);
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
};
