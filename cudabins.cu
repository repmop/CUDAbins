#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <json/json.h>
#include <jsoncpp.cpp>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <random>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "cudabins.h"

using namespace std;
using namespace thrust;

__constant__ cudaParams params;
__device__ cudaGlobals globals;

obj *host_objs;
host_vector<bin> host_bins;


uint32_t host_num_objs;
uint32_t host_bin_size;
uint32_t host_total_obj_size = 0; //pseudo-constant
uint32_t host_num_bins;

bin *bins_out;

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
    host_objs = new obj[host_num_objs];
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
void check_bin(dev_bin *b, int bin_size) {
    uint32_t sum = 0;
    for (int i = 0; i < b->obj_list.size(); i++) {
        sum += b->obj_list.arr[i].size;
    }
    assert(b->occupancy == sum);
    assert(b->occupancy <= bin_size);
}

void host_check_bin(bin *b, size_t bin_size) {
    uint32_t sum = 0;
    for (size_t i = 0; i < b->obj_list.size(); i++) {
        sum += b->obj_list[i].size;
    }
    assert(b->occupancy == sum);
    assert(b->occupancy <= bin_size);
}

// Select a random bin weighted in favor of empty bins. Uses data structures
//  generated in setup_rand, which may be stale.
__device__
uint32_t rand_empty(){
    ghetto_vec<float> ecdfs = globals.ecdfs;
    return ecdfs.upper_bound(globals.rand_f() / RAND_MAX - ecdfs[0]);
}

// Select a random bin weighted in favor of full bins.
__device__
uint32_t rand_full(){
    ghetto_vec<float> fcdfs = globals.fcdfs;
    return fcdfs.upper_bound(globals.rand_f() / RAND_MAX - fcdfs[0]);
}

// Recalculate data structures used by rand_empty & _full based on current bins.
__device__
void setup_rand(){
    int num_bins = globals.num_bins;
    dev_bin *bins = globals.bins;
    ghetto_vec<float> ecdfs = globals.ecdfs;
    ghetto_vec<float> fcdfs = globals.fcdfs;

    ecdfs.resize(num_bins);
    fcdfs.resize(num_bins);
    int bin_size = params.bin_size;
    int total_obj_size = params.total_obj_size;

    float sum_empty_space = (float)(num_bins * bin_size - total_obj_size);
    float ecdf = 0.f;
    float fcdf = 0.f;
    for(uint32_t i = 0; i < num_bins; i++){
        ecdf += ((float) (bin_size - bins[i].occupancy)) / sum_empty_space;
        ecdfs[i] = ecdf;
        fcdf += ((float) bins[i].occupancy) / total_obj_size;
        fcdfs[i] = fcdf;
    }
}

// Fill bins using next-fit
__device__
void runNF () {
    obj *objs = params.objs;
    int bin_size = params.bin_size;
    int num_objs = params.num_objs;
    int maxsize = params.maxsize;
    size_t *num_bins = &globals.num_bins;

    dev_bin *bins = globals.bins;

    // Start by allocating first bin
    size_t bi = 0; // Index of last valid bin
    dev_bin *b = &(bins[bi]);

    for (size_t oi = 0; oi < num_objs; oi++) {
        obj *o = &objs[oi];
        if(b->occupancy + o->size > bin_size){
            // Move to a new bin (space is already allocated)
            bi++;
            if (bi >= maxsize) {
                *num_bins = maxsize;
                return;
            }

            b = &(bins[bi]);
            *b = dev_bin(0);
        }
        b->obj_list.push_back(*o);
    }

    *num_bins = bi + 1;

    return;
}

__global__ void
kernelWalkPack() {

    int bin_size = params.bin_size;
    int maxsize = params.maxsize;
    int *dev_retval_pt = params.dev_retval_pt;
    obj *obj_out = params.obj_out;
    size_t *idx_out = params.idx_out;

    globals = cudaGlobals(maxsize);

    dev_bin *bins = globals.bins;
    size_t num_bins = globals.num_bins;

    // Start with next fit
    runNF();

    // This represents just one trial
    const int passes = 500;
    for(int pass = 0; pass < passes; pass++){
        // Optimize
        if (num_bins <= 1) {
            break;
        }

        size_t src;
        do {
            src = globals.rand_i() % num_bins;
        } while (!bins[src].valid);
        dev_bin *srcbin = &bins[src];

        for(uint32_t i = 0; i < srcbin->obj_list.size(); i++){
            uint32_t obj_size = srcbin->obj_list.arr[i].size;

            // Choose a destination bin that is not the src and has enough space
            size_t dest;
            const int retries = 1000; // How many destinations to try

            for(int j = 0; j < retries; j++){
                // Choose a destination other than the src
                while(src == (dest = globals.rand_i() % num_bins));
                // while(src == (dest = rand_full()));

                if(bins[dest].occupancy + obj_size <= bin_size){
                    break;
                }
            }
            dev_bin *destbin = &bins[dest];

            destbin->obj_list.push_back(srcbin->obj_list.arr[i]);
            destbin->occupancy += obj_size;
        }

        // Delete srcbin
        // bins.erase(bins.begin() + src);
        // srcbin->valid = false;
        for(size_t i = src; i < num_bins; i++){
            bins[i] = bins[i+1];
        }
        num_bins--;

        // TODO: only constrain the bins that need constraining
        // if(src < dest){
        //     dest--;
        // }
        // if(bins[dest].occupancy > bin_size){
        //     overflow_count++;
        //     constrain_bin(dest);
        // }

        // Constrain all bins

        // Do want i to track the growing bins list here
        for (size_t i = 0; i < num_bins; i++) {
            dev_bin *bin = &bins[i];
            dev_bin *newbin;

            // If bin is overfull, allocate a new bin
            if (bin->occupancy > bin_size) {
                if (num_bins >= maxsize - 1) {
                    *dev_retval_pt = -1;
                    printf("Too many bins\n");
                    return;
                }

                newbin = &bins[num_bins];
                num_bins++;
                *newbin = dev_bin(0);
            }

            // Move objects from overfull bin to new bin
            while (bin->occupancy > bin_size) {
                uint32_t r = globals.rand_i() % bin->obj_list.size();
                obj obj = bin->obj_list.arr[r];
                bin->obj_list.erase(r);
                bin->occupancy -= obj.size;
                newbin->occupancy += obj.size;
                newbin->obj_list.push_back(obj);
            }
        }
    }

    // TODO: Reduce across threads to find best packing

    // Copy objects in order to obj_out
    // idx_out holds the indices into obj_out where each bin starts
    size_t out_idx = 0;
    size_t bi;
    for(bi = 0; bi < num_bins; bi++){
      idx_out[bi] = out_idx;
      for(size_t oi = 0; oi < bins[bi].obj_list.size(); oi++){
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
kernel() {
    int bin_size = params.bin_size;
    int num_objs = params.num_objs;
    int maxsize = params.maxsize;
    int *dev_retval_pt = params.dev_retval_pt;
    obj *objs = params.objs;
    obj *obj_out = params.obj_out;
    size_t *idx_out = params.idx_out;

    globals = cudaGlobals(maxsize);

    dev_bin *bins = globals.bins;
    size_t num_bins = globals.num_bins;

    thrust::sort(cuda::par, objs, &objs[num_objs],
        [](const obj &a, const obj &b) -> bool { return a.size > b.size; });

    for (size_t i = 0; i < num_objs; i++) {
        obj obj = objs[i];
        bool found_fit_flag = false;
        for (size_t j = 0; j < num_bins; j++) {
            dev_bin *bin = &bins[j];
            if (bin->occupancy + obj.size <= bin_size) {
                bin->occupancy += obj.size;
                bin->obj_list.push_back(obj);
                found_fit_flag = true;
                break;
            }
            check_bin(bin, bin_size);
        }
        if (!found_fit_flag) {
            dev_bin b;
            b.occupancy = obj.size;
            if (num_bins >= maxsize - 1) {
                *dev_retval_pt = -1;
                return;
            }
            b.obj_list.push_back(obj);
            bins[num_bins] = b;
            num_bins++;
        }
    }

    // Copy objects to serial output
    size_t out_idx = 0;
    size_t bi;
    for(bi = 0; bi < num_bins; bi++){
      idx_out[bi] = out_idx;
      for(size_t oi = 0; oi < bins[bi].obj_list.size(); oi++){
        obj_out[out_idx] = bins[bi].obj_list.arr[oi];
        out_idx++;
      }
    }
    idx_out[bi] = out_idx;

    for (size_t j = 0; j < num_bins; j++) {
        check_bin(&bins[j], bin_size);
    }
    printf("Num bins: %zu\n", num_bins);
    *dev_retval_pt = (int) num_bins;
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
    obj *obj_out;
    size_t *idx_out;
    int *dev_retval_pt, host_retval, maxsize;
    cudaParams p;

    maxsize = calculate_maxsize();
    cout << "Max number of bins " << maxsize << std::endl;

    p.total_obj_size = host_total_obj_size;
    p.bin_size = host_bin_size;
    p.num_objs = host_num_objs;
    p.maxsize = maxsize;
    gpuErrchk(cudaMalloc(&p.dev_retval_pt, sizeof(int)));
    gpuErrchk(cudaMalloc(&p.objs, host_num_objs * sizeof(obj)));
    gpuErrchk(cudaMalloc(&obj_out, host_num_objs * sizeof(obj)));
    gpuErrchk(cudaMalloc(&idx_out, host_num_objs * sizeof(size_t)));
    dev_retval_pt = p.dev_retval_pt;

    gpuErrchk(cudaMemcpyToSymbol(params, &p, sizeof(cudaParams)));

    kernel<<<1,1>>>();
    cudaThreadSynchronize();

    cudaMemcpy(&host_retval, dev_retval_pt, sizeof(int), cudaMemcpyDeviceToHost);
    if (host_retval < 0) {
        cout << "CUDA kernel failed to pack bins\n";
    }
    host_num_bins = host_retval;
    bins_out = new bin[host_num_bins];

    // Copy the representation of objs in bins to host
    size_t *host_idxs = new size_t[host_num_bins+1];
    obj *host_objs = new obj[host_num_objs];
    cudaMemcpy(host_idxs, idx_out, (host_num_bins + 1) * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_objs, obj_out, (host_num_objs) * sizeof(obj),
               cudaMemcpyDeviceToHost);

    for (size_t bi = 0; bi < host_num_bins; bi++) {
        bin *b = new bin;
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
        bin bin = bins_out[i];
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
