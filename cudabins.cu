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

__inline__ __device__
void init_ghetto_vector(ghetto_vec_t *v, size_t size){
    v->maxlen = size;
    v->arr = new obj_t[v->maxlen];
    v->num_entries = 0;
}

typedef struct dev_bin {
  uint32_t occupancy;
  ghetto_vec obj_list;
  alias_t alias;
} dev_bin_t;

__inline__ __device__
void init_dev_bin(dev_bin_t *b, size_t obj_arr_size){
    init_ghetto_vector(&(b->obj_list), obj_arr_size);
    b->occupancy = 0;
    //b->alias = ___;
}

__inline__ __device__
void add_obj(dev_bin_t *bin, obj_t *obj){
    bin->occupancy += obj->size;
    bin->obj_list.push_back(*obj);
}

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

void host_check_bin(bin_t *b, size_t bin_size) {
    uint32_t sum = 0;
    for (size_t i = 0; i < b->obj_list.size(); i++) {
        sum += b->obj_list[i].size;
    }
    assert(b->occupancy == sum);
    assert(b->occupancy <= bin_size);
}


// Fill bins using next-fit
__device__
void runNF (dev_bin_t *bins, obj_t *objs, int num_objs, int bin_size,
            int maxsize, int *num_bins) {
    //bins.clear();

    // Start by allocating first bin
    size_t bi = 0; // Index of last valid bin
    dev_bin_t *b = &(bins[bi]);
    init_dev_bin(b, 10);

    for (size_t oi = 0; oi < num_objs; oi++) {
        obj_t *o = &objs[oi];
        if(b->occupancy + o->size > bin_size){
            // Move to a new bin (space is already allocated)
            bi++;
            if (bi >= maxsize) {
                *num_bins = -1;
                return;
            }

            b = &(bins[bi]);
            init_dev_bin(b, 10);
        }
        add_obj(b, o);
    }

    *num_bins = bi + 1;

    return;
}


__global__ void
kernelBFD(dev_bin_t *bins, int maxsize, int *dev_retval_pt,
          obj_t *obj_out, size_t *idx_out) {

    int num_bins = 0;
    int num_objs = params.num_objs;
    obj_t *objs = params.objs;
    int bin_size = params.bin_size;

    // Sort objects decreasing
    thrust::sort(cuda::par, objs, &objs[num_objs],
        [](const obj_t &a, const obj_t &b) -> bool { return a.size > b.size; });

    // Put each object in the first bin it fits into
    for (size_t i = 0; i < num_objs; i++) {
        obj_t obj = objs[i];

        // Scan through existing bins for space
        bool found_fit_flag = false;
        for (size_t j = 0; j < num_bins; j++) {
            dev_bin_t *bin = &bins[j];
            if (bin->occupancy + obj.size <= bin_size) {
                add_obj(bin, &obj);
                found_fit_flag = true;
                break;
            }
            check_bin(bin, bin_size);
        }

        // If you don't find any, make a new bin
        if (!found_fit_flag) {
            dev_bin_t b;
            if (num_bins >= maxsize - 1) {
                *dev_retval_pt = -1;
                return;
            }

            // Make a new bin and put the object in it
            init_dev_bin(&b, 10);
            add_obj(&b, &obj);

            // Add this bin to bins
            bins[num_bins] = b;
            num_bins++;
        }
    }

    // Copy objects in order to obj_out
    // idx_out holds the indices into obj_out where each bin starts
    size_t out_idx = 0;
    size_t bi;
    for(bi = 0; bi < num_bins; bi++){
      idx_out[bi] = out_idx;
      for(size_t oi = 0; oi < bins[bi].obj_list.num_entries; oi++){
        obj_out[out_idx] = bins[bi].obj_list.arr[oi];
        out_idx++;
      }
    }
    idx_out[bi] = out_idx;

    for (size_t j = 0; j < num_bins; j++) {
        check_bin(&bins[j], bin_size);
    }

    // Return the number of bins
    *dev_retval_pt = num_bins;
    printf("Num bins: %i\n", num_bins);

    return;
}

__global__ void
kernelWalkPack(dev_bin_t *bins, int maxsize, int *dev_retval_pt,
               obj_t *obj_out, size_t *idx_out) {

    int num_bins;
    int num_objs = params.num_objs;
    obj_t *objs = params.objs;
    int bin_size = params.bin_size;

    // Pre-order
    runNF(bins, objs, num_objs, bin_size, maxsize, &num_bins);

    // Copy objects in order to obj_out
    // idx_out holds the indices into obj_out where each bin starts
    size_t out_idx = 0;
    size_t bi;
    for(bi = 0; bi < num_bins; bi++){
      idx_out[bi] = out_idx;
      for(size_t oi = 0; oi < bins[bi].obj_list.num_entries; oi++){
        obj_out[out_idx] = bins[bi].obj_list.arr[oi];
        out_idx++;
      }
    }
    idx_out[bi] = out_idx;

    for (size_t j = 0; j < num_bins; j++) {
        check_bin(&bins[j], bin_size);
    }

    // Return the number of bins
    *dev_retval_pt = num_bins;
    printf("Num bins: %i\n", num_bins);

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
void runBFD() {
    // Inputs
    dev_bin_t *bins;
    obj_t *objs;

    // Outputs
    obj_t *obj_out;
    size_t *idx_out;

    // Calculate a high water mark for number of bins used
    int maxsize = calculate_maxsize();
    cout << "Max number of bins " << maxsize << std::endl;

    // Allocate space on device
    int *dev_retval_pt, host_retval;
    gpuErrchk(cudaMalloc(&dev_retval_pt, sizeof(int)));
    gpuErrchk(cudaMalloc(&bins, maxsize * sizeof(dev_bin_t)));
    gpuErrchk(cudaMalloc(&objs, host_num_objs * sizeof(obj_t)));
    gpuErrchk(cudaMalloc(&obj_out, host_num_objs * sizeof(obj_t)));
    gpuErrchk(cudaMalloc(&idx_out, host_num_objs * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(objs, host_objs, host_num_objs * sizeof(obj_t),
                         cudaMemcpyHostToDevice));

    // Assign parameters for device
    cudaParams p;
    p.objs = objs;
    p.num_objs = host_num_objs;
    p.bin_size = host_bin_size;
    p.total_obj_size = host_total_obj_size;
    gpuErrchk(cudaMemcpyToSymbol(params, &p, sizeof(cudaParams)));

    // Run BFD
    kernelBFD<<<1,1>>>(raw_pointer_cast(&bins[0]), maxsize, dev_retval_pt,
                       obj_out, idx_out);
    cudaThreadSynchronize();

    // Copy back number of bins
    cudaMemcpy(&host_retval, dev_retval_pt, sizeof(int), cudaMemcpyDeviceToHost);
    if (host_retval < 0) {
        cout << "CUDA kernel failed to pack bins\n";
    }
    host_num_bins = host_retval;
    bins_out = new bin_t[host_num_bins];

    // Copy the representation of objs in bins to host
    size_t *host_idxs = new size_t[host_num_bins+1];
    obj_t *host_objs = new obj_t[host_num_objs];
    cudaMemcpy(host_idxs, idx_out, (host_num_bins + 1) * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_objs, obj_out, (host_num_objs) * sizeof(obj_t),
               cudaMemcpyDeviceToHost);

    // Create a vector with these objects
    for (size_t bi = 0; bi < host_num_bins; bi++) {
        bin_t *b = new bin_t;
        for(size_t oi = host_idxs[bi]; oi < host_idxs[bi + 1]; oi++){
            b->obj_list.push_back(host_objs[oi]);
        }
        bins_out[bi] = *b;
    }
}

__host__
void run() {
    // Inputs
    dev_bin_t *bins;
    obj_t *objs;

    // Outputs
    obj_t *obj_out;
    size_t *idx_out;

    // Calculate a high water mark for number of bins used
    int maxsize = calculate_maxsize();
    cout << "Max number of bins " << maxsize << std::endl;

    // Allocate space on device
    int *dev_retval_pt, host_retval;
    gpuErrchk(cudaMalloc(&dev_retval_pt, sizeof(int)));
    gpuErrchk(cudaMalloc(&bins, maxsize * sizeof(dev_bin_t)));
    gpuErrchk(cudaMalloc(&objs, host_num_objs * sizeof(obj_t)));
    gpuErrchk(cudaMalloc(&obj_out, host_num_objs * sizeof(obj_t)));
    gpuErrchk(cudaMalloc(&idx_out, host_num_objs * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(objs, host_objs, host_num_objs * sizeof(obj_t),
                         cudaMemcpyHostToDevice));

    // Assign parameters for device
    cudaParams p;
    p.objs = objs;
    p.num_objs = host_num_objs;
    p.bin_size = host_bin_size;
    p.total_obj_size = host_total_obj_size;
    gpuErrchk(cudaMemcpyToSymbol(params, &p, sizeof(cudaParams)));

    // Run BFD
    kernelWalkPack<<<1,1>>>(raw_pointer_cast(&bins[0]), maxsize, dev_retval_pt,
                            obj_out, idx_out);
    cudaThreadSynchronize();

    // Copy back number of bins
    cudaMemcpy(&host_retval, dev_retval_pt, sizeof(int), cudaMemcpyDeviceToHost);
    if (host_retval < 0) {
        cout << "CUDA kernel failed to pack bins\n";
    }
    host_num_bins = host_retval;
    bins_out = new bin_t[host_num_bins];

    // Copy the representation of objs in bins to host
    size_t *host_idxs = new size_t[host_num_bins+1];
    obj_t *host_objs = new obj_t[host_num_objs];
    cudaMemcpy(host_idxs, idx_out, (host_num_bins + 1) * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_objs, obj_out, (host_num_objs) * sizeof(obj_t),
               cudaMemcpyDeviceToHost);

    // Create a vector with these objects
    for (size_t bi = 0; bi < host_num_bins; bi++) {
        bin_t *b = new bin_t;
        for(size_t oi = host_idxs[bi]; oi < host_idxs[bi + 1]; oi++){
          b->obj_list.push_back(host_objs[oi]);
        }
        bins_out[bi] = *b;
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
