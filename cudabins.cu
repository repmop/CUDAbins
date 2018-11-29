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
#define SCALE 2

typedef struct obj {
    uint32_t size;
} obj_t;

typedef struct alias {
    int i1;
    int i2;
    float divider;
} alias_t;

typedef struct ghetto_vec {
    __device__
    void push_back(obj_t obj) {
        if (num_entries + 1 >= maxlen) {
            obj_t *old = arr;
            arr = new obj_t[maxlen * SCALE];
            for (int i = 0; i < num_entries; i++) {
                arr[i] = old[i];
            }
            delete[] old;
        }
        arr[num_entries++] = obj;
    }
    __device__
    int size() {
        return num_entries;
    }
    int maxlen;
    int num_entries;
    obj_t *arr;
} ghetto_vec_t;

typedef struct dev_bin {
  uint32_t occupancy;
  ghetto_vec obj_list;
  alias_t alias;
} dev_bin_t;

typedef struct bin {
  uint32_t occupancy;
  host_vector<obj_t> obj_list;
  alias_t alias;
  __host__ ~bin() {}
} bin_t;

typedef struct cudaParams {
    uint32_t total_obj_size;
    uint32_t bin_size;
    uint32_t num_objs;
    obj_t *objs;
} cudaParams;

obj_t *objs;
__constant__ cudaParams params;

obj_t *host_objs;
host_vector<bin_t> host_bins;
uint32_t host_num_objs;
uint32_t host_bin_size;
uint32_t host_total_obj_size = 0; //pseudo-constant
uint32_t host_num_bins;

bin_t *bins_out;

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

__host__
int calculate_maxsize() {
    const float slip_ratio = .5f;
    int total_size = 0;
    for (size_t i = 0; i < host_num_objs; i++) {
        total_size += host_objs[i].size;
    }
    return (int) ((float) total_size / (slip_ratio * host_bin_size));
}
__device__
void check_bin(dev_bin_t *b, int bin_size) {
    uint32_t sum = 0;
    for (int i = 0; i < b->obj_list.num_entries; i++) {
        sum += b->obj_list.arr[i].size;
    }
    assert(b->occupancy == sum);
    assert(b->occupancy <= bin_size);
}

__global__ void
kernel(dev_bin_t *bins, int maxsize, int *dev_retval_pt) {
    int size = 0;
    int num_objs = params.num_objs;
    obj_t *objs = params.objs;
    int bin_size = params.bin_size;
    // int total_obj_size = params.total_obj_size;
    // printf("bins: %p, maxsize: %i, dev_retval_pt: %p\n",bins,maxsize,dev_retval_pt);
    // printf("num_objs: %i, objs: %p, bin_size: %i, total_obj_size: %i\n",num_objs,objs,bin_size,total_obj_size);
    thrust::sort(cuda::par, objs, &objs[num_objs],
        [](const obj_t &a, const obj_t &b) -> bool { return a.size > b.size; });
    for (size_t i = 0; i < num_objs; i++) {
        obj_t obj = objs[i];
        bool found_fit_flag = false;
        for (size_t j = 0; j < size; j++) {
            dev_bin_t *bin = &bins[j];
            if (bin->occupancy + obj.size <= bin_size) {
                bin->occupancy += obj.size;
                bin->obj_list.push_back(obj);
                found_fit_flag = true;
                break;
            }
            check_bin(bin, bin_size);
        }
        if (!found_fit_flag) {
            dev_bin_t b;
            b.occupancy = obj.size;
            if (size >= maxsize - 1) {
                *dev_retval_pt = -1;
                return;
            }
            b.obj_list.maxlen = 10;
            b.obj_list.arr = new obj_t[b.obj_list.maxlen];
            b.obj_list.num_entries = 0;
            b.obj_list.push_back(obj);
            bins[size] = b;
            size++;
        }
    }
    for (size_t j = 0; j < size; j++) {
        check_bin(&bins[j], bin_size);
    }
    printf("size: %i\n", size);
    *dev_retval_pt = size;
    return;
}

void runBFD(){
    return;
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__
void run() {
    dev_bin_t *bins;
    int maxsize = calculate_maxsize();
    int *dev_retval_pt, host_retval;
    cudaParams p;
    gpuErrchk(cudaMalloc(&dev_retval_pt, sizeof(int)));
    gpuErrchk(cudaMalloc(&bins, maxsize * sizeof(dev_bin_t)));
    gpuErrchk(cudaMalloc(&objs, host_num_objs * sizeof(obj_t)));
    gpuErrchk(cudaMemcpy(objs, host_objs, host_num_objs * sizeof(obj_t), cudaMemcpyHostToDevice));
    p.objs = objs;
    p.num_objs = host_num_objs;
    p.bin_size = host_bin_size;
    p.total_obj_size = host_total_obj_size;
    gpuErrchk(cudaMemcpyToSymbol(params, &p, sizeof(cudaParams)));

    kernel<<<1,1>>>(raw_pointer_cast(&bins[0]), maxsize, dev_retval_pt);
    cudaThreadSynchronize();

    cudaMemcpy(&host_retval, dev_retval_pt, sizeof(int), cudaMemcpyDeviceToHost);
    if (host_retval < 0) {
        cout << "CUDA kernel failed to pack bins\n";
    }
    host_num_bins = host_retval;
    bins_out = new bin_t[host_num_bins];
    for (size_t i = 0; i < host_num_bins; i++) {
        int objs_in_bin;
        cudaMemcpy(&objs_in_bin, &bins[i].obj_list.num_entries,
                   sizeof(int), cudaMemcpyDeviceToHost);
        bin_t *b = new bin_t;
        b->obj_list.resize(objs_in_bin);

        cudaMemcpy(&b->obj_list[0], &bins[i].obj_list.arr,
                   objs_in_bin * sizeof(obj_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b->occupancy, &bins[i].occupancy,
                   sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b->alias, &bins[i].alias,
                   sizeof(alias_t), cudaMemcpyDeviceToHost);
        bins_out[i] = *b;
    }
}
__host__
bool dump(char *outfile) {
    Json::Value obj_data;
    obj_data["bin_size"] = host_bin_size;
    obj_data["num_objs"] = host_num_objs;
    obj_data["num_bins"] = host_num_bins;
    obj_data["objs"] = Json::Value(Json::arrayValue);
    obj_data["bins"] = Json::Value(Json::arrayValue);
    for(uint32_t i = 0; i < host_num_objs; i++){
        obj_data["objs"][i] = host_objs[i].size;
    }
    for(uint32_t i = 0; i < (uint32_t) host_num_bins; i++) {
        bin_t bin = bins_out[i];
        obj_data["bins"][i] = Json::Value(Json::arrayValue);
        for (uint32_t j = 0; j < bin.obj_list.size(); j++) {
            obj_data["bins"][i][j] = bin.obj_list[j].size;
        }
    }
    if (outfile==NULL) { //print results to stdout
        cout << "num_objs: " << host_num_objs << endl;
        cout << "num_bins: " << host_num_bins << endl;
    } else { //print to file
        filebuf fb;
        fb.open(outfile, ios::out);
        ostream f(&fb);
        f << obj_data;
    }
    delete[] host_objs;
    return true;
}
