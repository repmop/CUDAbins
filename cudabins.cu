#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <json/json.h>
#include <assert.h>
#include <jsoncpp.cpp>
#include <iostream>
#include <fstream>
#include <random>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

using namespace std;
using namespace thrust;

#define BFD


typedef struct obj {
    uint32_t size;
} obj_t;

typedef struct alias {
    int i1;
    int i2;
    float divider;
} alias_t;

typedef struct bin {
  uint32_t occupancy;
  alias_t alias;
} bin_t;


__constant__ uint32_t total_obj_size = 0; //pseudo-constant
__constant__ uint32_t bin_size;
__constant__ uint32_t num_objs;
__device__ obj_t *objs;

obj_t *host_objs;
host_vector<bin_t> host_bins;
uint32_t host_num_objs;
uint32_t host_bin_size;
uint32_t host_total_obj_size = 0; //pseudo-constant


host_vector<bin_t> bins_out;

__host__
bool parse(char *infile) {
    ifstream f(infile, std::ifstream::binary);
    if (f.fail()) {
        return false;
    }
    Json::Value obj_data;
    f >> obj_data;
    host_bin_size = obj_data["bin_size"].asUInt();
    host_num_objs = obj_data["num_objs"].asUInt();

    // Initialize object array and put one obj in each bin
    host_objs = new obj_t[host_num_objs];
    auto obj_array = obj_data["objs"];
    for(uint32_t i = 0; i < host_num_objs; i++){
        #ifdef TAGGING
        host_objs[i].tag = i;
        #endif
        host_objs[i].size = obj_array[i].asUInt();
        host_total_obj_size += obj_array[i].asUInt();

    }
    return true;
}

__global__ void
kernel(bin_t *bins, int size) {
    thrust::sort(cuda::par, objs, &objs[num_objs],
        [](const obj_t &a, const obj_t &b) -> bool { return a.size > b.size; });
    for (size_t i = 0; i < num_objs; i++) {
        obj_t obj = objs[i];
        bool found_fit_flag = false;
        for (size_t j = 0; j < size; j++) {
            bin_t *bin = &bins[j];
            if (bin->occupancy + obj.size <= bin_size) {
                bin->occupancy += obj.size;
                // bin->obj_list.push_back(obj);
                found_fit_flag = true;
                break;
            }
        }
        if (!found_fit_flag) {
            // bins.push_back(*make_bin(&obj));
            bin_t b;
            b.occupancy = obj.size;
            bins[size] = b;
            size++;
        }
    }
    return;
}

void runBFD(){
    return;
}

__host__
void run() {
    device_vector<bin_t> bins;
    bins = host_bins;
    cudaMemcpy(&host_objs, &objs, host_num_objs * sizeof(obj_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&host_num_objs, &num_objs, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&host_bin_size, &bin_size, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&host_total_obj_size, &total_obj_size, sizeof(uint32_t), cudaMemcpyHostToDevice);
    kernel<<<1,1>>>(raw_pointer_cast(&bins[0]), bins.size());
    cudaThreadSynchronize();
    bins_out = bins;
}
__host__
bool dump(char *outfile) {
    Json::Value obj_data;
    obj_data["bin_size"] = host_bin_size;
    obj_data["num_objs"] = host_num_objs;
    obj_data["num_bins"] = bins_out.size();
    obj_data["objs"] = Json::Value(Json::arrayValue);
    obj_data["bins"] = Json::Value(Json::arrayValue);
    for(uint32_t i = 0; i < host_num_objs; i++){
        obj_data["objs"][i] = host_objs[i].size;
    }
    if (outfile==NULL) { //print results to stdout
        cout << "num_objs: " << host_num_objs << endl;
        cout << "num_bins: " << bins_out.size() << endl;
    } else { //print to file
        filebuf fb;
        fb.open(outfile, ios::out);
        ostream f(&fb);
        f << obj_data;
    }
    delete[] host_objs;
    return true;
}
