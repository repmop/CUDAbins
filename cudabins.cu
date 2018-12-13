#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>


#include <assert.h>

#include <random>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "cudabins.h"

using namespace std;
using namespace thrust;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__constant__ cudaParams params;
__device__ cudaGlobals globals;


uint32_t host_num_bins;

bin *bins_out;

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
    printf("Starting NF\n");

    obj *objs = params.objs;
    int bin_size = params.bin_size;
    int num_objs = params.num_objs;
    int maxsize = params.maxsize;
    size_t *num_bins = &globals.num_bins;

    dev_bin *bins = globals.bins;

    // Start by allocating first bin
    size_t bi = 0; // Index of last valid bin
    dev_bin *b = &(bins[bi]);

    for(size_t oi = 0; oi < num_objs; oi++){
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
        b->occupancy += o->size;
    }

    *num_bins = bi + 1;
    printf("NF done with %lu bins\n", *num_bins);
    return;
}


__global__ void
kernelBFD() {
    int bin_size = params.bin_size;
    int num_objs = params.num_objs;
    int maxsize = params.maxsize;
    int *dev_retval_pt = params.dev_retval_pt;
    obj *objs = params.objs;
    obj *obj_out = params.obj_out;
    size_t *idx_out = params.idx_out;

    globals = cudaGlobals(maxsize);
    setup_rand();

    dev_bin *bins = globals.bins;
    size_t num_bins = globals.num_bins;

    // Sort objects decreasing
    thrust::sort(cuda::par, objs, &objs[num_objs],
        [](const obj &a, const obj &b) -> bool { return a.size > b.size; });

    // Put each object in the first bin it fits into
    for (size_t i = 0; i < num_objs; i++) {
        obj obj = objs[i];
        // Scan through existing bins for space
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

        // If you don't find any, make a new bin
        if (!found_fit_flag) {
            if (num_bins >= maxsize - 1) {
                // printf("Mysterious\n");
                *dev_retval_pt = -1;
                return;
            }

            // Make a new bin and put the object in it
            dev_bin b = dev_bin(0);
            b.occupancy = obj.size;
            b.obj_list.push_back(obj);

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
    *dev_retval_pt = (int) num_bins;
    printf("Finished, Num bins: %i\n", num_bins);

    return;
}


__global__ void
kernelWalkPack() {
    // ID of thread within this trial. Trials are mapped to blocks,
    //  so thread_id does not depend on blockIdx
    int thread_id = threadIdx.x;

    int bin_size = params.bin_size;
    int maxsize = params.maxsize;
    int *num_bins = params.dev_retval_pt; // Managed memory for atomic add

    obj *obj_out = params.obj_out;
    size_t *idx_out = params.idx_out;

    if(thread_id == 0){
        // TODO: will this work?
        globals = cudaGlobals(maxsize);
    }
    // TODO: necessary? unlikely
    __syncthreads();

    dev_bin *bins = globals.bins;

    // Start with next fit
    if(thread_id == 0){
        runNF();
    }

    // A little hacky
    *num_bins = globals.num_bins; // TODO: update globals

    if(*num_bins >= maxsize){
        *num_bins = -1;
        if(thread_id == 0)
            printf("Next Fit failed\n");

        return;
    }

    __syncthreads();

    // This represents just one trial
    const int passes = 400;
    for(int pass = 0; pass < passes; pass++){

        // Optimize
        if (*num_bins <= 1) {
            printf("breaking\n");
            break;
        }

        size_t src;
        do {
            src = globals.rand_i() % *num_bins;
        } while (!bins[src].valid);
        dev_bin *srcbin = &bins[src];

        if(thread_id == 0){
            for(uint32_t i = 0; i < srcbin->obj_list.size(); i++){
                uint32_t obj_size = srcbin->obj_list.arr[i].size;

                // Choose a destination bin that is not the src and has enough space
                size_t dest;
                const int retries = 1000; // How many destinations to try

                for(int j = 0; j < retries; j++){
                    // Choose a destination other than the src
                    while(src == (dest = globals.rand_i() % *num_bins));
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
            for(size_t i = src; i < *num_bins; i++){
                bins[i] = bins[i+1];
            }
            (*num_bins)--;
        }
        __syncthreads();

        // TODO: only constrain the bins that need constraining
        // if(src < dest){
        //     dest--;
        // }
        // if(bins[dest].occupancy > bin_size){
        //     overflow_count++;
        //     constrain_bin(dest);
        // }

        // Constrain

        size_t bins_per_thread = (*num_bins + blockDim.x - 1) / blockDim.x;
        size_t start_bin = thread_id * bins_per_thread;
        size_t end_bin = (thread_id + 1) * bins_per_thread;
        size_t old_num_bins = *num_bins; // Freeze num_bins for the for loop

        // Constrain all the old bins
        // All new bins must not be overfull
        for (size_t i = start_bin; i < end_bin && i < old_num_bins; i++) {
            dev_bin *bin = &bins[i];
            dev_bin *newbin;

            // If bin is overfull, allocate a new bin
            while (bin->occupancy > bin_size) {

                // Requires atomic add, but very low contention
                size_t new_bin_idx = atomicAdd(num_bins, 1);

                if (new_bin_idx >= maxsize) {
                    *num_bins = -1;
                    printf("Too many bins\n");
                    return;
                }

                newbin = &bins[new_bin_idx];
                *newbin = dev_bin(0);

                // Move objects from overfull bin to new bin
                while (bin->occupancy > bin_size) {
                    uint32_t r = globals.rand_i() % bin->obj_list.size();
                    obj obj = bin->obj_list.arr[r];
                    bin->obj_list.erase(r);
                    bin->occupancy -= obj.size;
                    newbin->occupancy += obj.size;
                    newbin->obj_list.push_back(obj);
                }

                // Make this swap so the next iteration of the while loop can
                //  constrain the new bin if it needs
                bin = newbin;
            }
        }
    }

    // TODO: Reduce across threads to find best packing

    // Copy objects in order to obj_out
    // idx_out holds the indices into obj_out where each bin starts
    if(thread_id == 0){
        size_t out_idx = 0;
        size_t bi;
        for(bi = 0; bi < *num_bins; bi++){
            idx_out[bi] = out_idx;
            for(size_t oi = 0; oi < bins[bi].obj_list.size(); oi++){
                obj_out[out_idx] = bins[bi].obj_list.arr[oi];
                out_idx++;
            }
        }
        idx_out[bi] = out_idx;

        for (size_t j = 0; j < *num_bins; j++) {
            check_bin(&bins[j], bin_size);
        }

        // Return the number of bins
        //*dev_retval_pt = num_bins;
        printf("Num bins: %i\n", *num_bins);
    }

    __syncthreads();
    return;
}


void runBFD(){
    //Inputs
    cudaParams p;

    // Outputs
    obj *obj_out;
    size_t *idx_out;
    int *dev_retval_pt, host_retval, maxsize;

    // Calculate a high water mark for number of bins used
    maxsize = calculate_maxsize();
    cout << "Max number of bins " << maxsize << std::endl;

    // Assign parameters for device
    p.total_obj_size = host_total_obj_size;
    p.bin_size = host_bin_size;
    p.num_objs = host_num_objs;
    p.maxsize = maxsize;

    // Allocate space on device
    gpuErrchk(cudaMalloc(&p.dev_retval_pt, sizeof(int)));
    gpuErrchk(cudaMalloc(&p.objs, host_num_objs * sizeof(obj)));
    gpuErrchk(cudaMalloc(&p.obj_out, host_num_objs * sizeof(obj)));
    gpuErrchk(cudaMalloc(&p.idx_out, host_num_objs * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(p.objs, host_objs, host_num_objs * sizeof(obj), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(params, &p, sizeof(cudaParams)));

    // Run BFD
    kernelBFD<<<1,1>>>();
    cudaThreadSynchronize();
    dev_retval_pt = p.dev_retval_pt;
    obj_out = p.obj_out;
    idx_out = p.idx_out;

    // Copy back number of bins
    gpuErrchk(cudaMemcpy(&host_retval, dev_retval_pt, sizeof(int), cudaMemcpyDeviceToHost));
    if (host_retval < 0) {
        cout << "CUDA kernel failed to pack bins\n";
    }
    host_num_bins = host_retval;
    bins_out = new bin[host_num_bins];

    // Copy the representation of objs in bins to host
    size_t host_idxs[host_num_bins+1];
    obj host_objs[host_num_objs];
    gpuErrchk(cudaMemcpy(host_idxs, idx_out, (host_num_bins + 1) * sizeof(size_t),
               cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(host_objs, obj_out, (host_num_objs) * sizeof(obj),
               cudaMemcpyDeviceToHost));

    // Create a vector with these objects
    for (size_t bi = 0; bi < host_num_bins; bi++) {
        bin  b;
        for(size_t oi = host_idxs[bi]; oi < host_idxs[bi + 1]; oi++){
          b.obj_list.push_back(host_objs[oi]);
        }
        bins_out[bi] = b;
    }
}


__host__
void run() {
    //Inputs
    cudaParams p;

    // Outputs
    obj *obj_out;
    size_t *idx_out;
    int *dev_retval_pt, host_retval, maxsize;

    // Calculate a high water mark for number of bins used
    maxsize = calculate_maxsize();
    cout << "Max number of bins " << maxsize << std::endl;

    // Assign parameters for device
    p.total_obj_size = host_total_obj_size;
    p.bin_size = host_bin_size;
    p.num_objs = host_num_objs;
    p.maxsize = maxsize;

    // Allocate space on device
    gpuErrchk(cudaMallocManaged(&p.dev_retval_pt, sizeof(int))); // Need managed memory for atomic add
    gpuErrchk(cudaMalloc(&p.objs, host_num_objs * sizeof(obj)));
    gpuErrchk(cudaMalloc(&p.obj_out, host_num_objs * sizeof(obj)));
    gpuErrchk(cudaMalloc(&p.idx_out, host_num_objs * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(p.objs, host_objs, host_num_objs * sizeof(obj), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(params, &p, sizeof(cudaParams)));


    // Run WalkPack
    int trials = 1;
    int threads_per_trial = 1;
    dim3 walkpack_block_dim(threads_per_trial, 1);
    dim3 walkpack_grid_dim (trials, 1);

    kernelWalkPack<<<walkpack_grid_dim, walkpack_block_dim>>>();
    cudaThreadSynchronize();
    dev_retval_pt = p.dev_retval_pt;
    obj_out = p.obj_out;
    idx_out = p.idx_out;

    // Copy back number of bins
    gpuErrchk(cudaMemcpy(&host_retval, dev_retval_pt, sizeof(int), cudaMemcpyDeviceToHost));
    if (host_retval < 0) {
        cout << "CUDA kernel failed to pack bins\n";
    }
    host_num_bins = host_retval;
    bins_out = new bin[host_num_bins];

    // Copy the representation of objs in bins to host
    size_t host_idxs[host_num_bins+1];
    obj host_objs[host_num_objs];
    gpuErrchk(cudaMemcpy(host_idxs, idx_out, (host_num_bins + 1) * sizeof(size_t),
               cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(host_objs, obj_out, (host_num_objs) * sizeof(obj),
               cudaMemcpyDeviceToHost));

    // Create a vector with these objects
    for (size_t bi = 0; bi < host_num_bins; bi++) {
        bin  b;
        for(size_t oi = host_idxs[bi]; oi < host_idxs[bi + 1]; oi++){
          b.obj_list.push_back(host_objs[oi]);
        }
        bins_out[bi] = b;
    }
}