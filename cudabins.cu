#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <assert.h>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#include "cudabins.h"

#define TRIALS 24

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__shared__ cudaGlobals globals;
__constant__ cudaParams params;
uint32_t host_num_bins;
bin *bins_out;
extern int threads_per_block;

__host__
int calculate_maxsize() {
    const float slip_ratio = .5f;
    int total_size = 0;
    for (size_t i = 0; i < host_num_objs; i++) {
        total_size += host_objs[i].size;
    }
    printf("Theoretical Minimum number of bins: %i\n",
           (total_size + host_bin_size - 1) / host_bin_size);
    return (int) ((float) total_size / (slip_ratio * host_bin_size));
}

__device__
void check_bin(dev_bin *b, int bin_size, int linum) {
    uint32_t sum = 0;
    for (int i = 0; i < b->obj_list.size(); i++) {
        sum += b->obj_list.arr[i].size;
    }

    char *temp = (char *) (b - globals.bins);
    size_t i = ((size_t) temp) / sizeof(dev_bin);
    if(b->occupancy != sum || b->occupancy > bin_size) {
        printf("Assert fail on line %d\n", linum);
        printf("Problem bin: %lu\n", i);
        printf("occupancy: %i | sum: %i | bin_size: %i\n",
               b->occupancy, sum, bin_size);
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

template<typename T>
__device__ __inline__
T clamp(T x, T lower, T upper) {
    if (x >= lower && x <= upper) {
        return x;
    }
    if (x < lower) {
        return lower;
    }
    return upper;
}

// Select a random bin weighted in favor of empty bins. Uses data structures
//  generated in setup_rand, which may be stale.
__device__
uint32_t rand_empty(){
    ghetto_vec<float> ecdfs = globals.ecdfs;
    float target = globals.rand_f();
    uint32_t ret = ecdfs.upper_bound(target);
    ret = clamp (ret, 0U, (uint32_t) globals.num_bins - 1);
    return ret;
}

// Select a random bin weighted in favor of full bins.
__device__
uint32_t rand_full(){
    ghetto_vec<float> fcdfs = globals.fcdfs;
    float target = globals.rand_f();
    uint32_t ret = fcdfs.upper_bound(target);
    ret = clamp (ret, 0U, (uint32_t) globals.num_bins - 1);
    return ret;
}

// Recalculate data structures used by rand_empty & _full based on current bins.
__device__
void setup_rand(){
    int num_bins = globals.num_bins;
    dev_bin *bins = globals.bins;
    ghetto_vec<float> &ecdfs = globals.ecdfs;
    ghetto_vec<float> &fcdfs = globals.fcdfs;

    ecdfs.resize(num_bins);
    fcdfs.resize(num_bins);
    ecdfs.num_entries = num_bins;
    fcdfs.num_entries = num_bins;
    int bin_size = params.bin_size;
    int total_obj_size = params.total_obj_size;
    float sum_empty_space = (float)(num_bins * bin_size - total_obj_size);

    fc_op thrust_operator(0); //need a dummy constructor

    ec_op thrust_operator_ec(bin_size);

    div_op ec_div(sum_empty_space);

    div_op fc_div(total_obj_size);

    thrust::exclusive_scan(thrust::cuda::par, bins, &bins[num_bins],
                           &fcdfs[0], 0.f, thrust_operator);
    thrust::exclusive_scan(thrust::cuda::par, bins, &bins[num_bins],
                           &ecdfs[0], 0.f, thrust_operator_ec);
    thrust::transform(thrust::cuda::par, &fcdfs[0], &fcdfs[num_bins],
                      &fcdfs[0], fc_div);
    thrust::transform(thrust::cuda::par, &ecdfs[0], &ecdfs[num_bins],
                      &ecdfs[0], ec_div);
}

// Fill bins using next-fit
__device__
void runNF () {
    printf("Starting NF\n");

    obj *objs = params.objs;
    int bin_size = params.bin_size;
    int num_objs = params.num_objs;
    int maxsize = params.maxsize;
    size_t *num_bins = &(globals.num_bins);

    dev_bin *bins = globals.bins;

    // Start by allocating first bin
    size_t bi = 0; // Index of last valid bin
    dev_bin *b = &(bins[bi]);
    *b = dev_bin(0);

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
    return;
}

__device__ __inline__
void my_tabulate(int *start, int n, bin_relu_op op) {
    if (n==0) {
        return;
    }
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    size_t ind_per_thread = (n + num_threads - 1) / num_threads;
    size_t start_ind = thread_id * ind_per_thread;
    size_t end_ind = (thread_id + 1) * ind_per_thread;
    if (end_ind > start_ind + n) {
        end_ind = start_ind + n;
    }
    if (thread_id >= n) {
        end_ind = start_ind;
    }
    for (size_t i = start_ind; i < end_ind; i++) {
        start[i] = op((int) i);
    }
}

__device__ __inline__
int my_min(int *start, int n) {
    if (n==0) {
        return INT_MAX;
    }
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    size_t ind_per_thread = (n + num_threads - 1) / num_threads;
    size_t start_ind = thread_id * ind_per_thread;
    size_t end_ind = (thread_id + 1) * ind_per_thread;
    if (end_ind > start_ind + n) {
        end_ind = start_ind + n;
    }
    if (thread_id >= n) {
        end_ind = start_ind;
    }
    __shared__ int *minima;
    if (thread_id == 0) {
        minima = new int[num_threads];
    }
    int private_min = INT_MAX;
    for (size_t i = start_ind; i < end_ind; i++) {
        if (start[i] < private_min) {
            private_min = start[i];
        }
    }
    __syncthreads();
    __shared__ int min;
    minima[thread_id] = private_min;
    __syncthreads();
    if (thread_id==0) {
        min = INT_MAX;
        for (size_t i = 0; i < num_threads; i++) {
            if (minima[i] < min) {
                min = minima[i];
            }
        }
        delete[] minima;
    }
    __syncthreads();
    return min;
}
__global__
void kernelBFD() {
    double sort_time, reduce_time, tab_time, start, t1;
    int thread_id = threadIdx.x;
    int bin_size = params.bin_size;
    int num_objs = params.num_objs;
    int maxsize = params.maxsize;
    int *dev_retval_pt = params.dev_retval_pt;
    obj *objs = params.objs;
    obj *obj_out = params.obj_out;
    size_t *idx_out = params.idx_out;

    __shared__ size_t num_bins;
    __shared__ int *indices;
    __shared__ dev_bin *bins;
    if(thread_id == 0){
        globals = cudaGlobals(maxsize);
        num_bins = 0;
        indices = new int[maxsize];
        bins = globals.bins;
        start = clock();
    }
    __syncthreads();
    // Sort objects decreasing
    if (thread_id == 0) {
        t1 = clock();
        thrust::sort(thrust::cuda::par, objs, &objs[num_objs],
            [](const obj &a, const obj &b) -> bool { return a.size > b.size; });
        sort_time = (clock() - t1);
    }

    // Put each object in the first bin it fits into
    for (size_t i = 0; i < num_objs; i++) {
        __syncthreads();
        obj *obj = &objs[i];

        bool found_fit_flag = false;

        // Parallel reduction to find minimum index that fits this bin
        bin_relu_op bin_relu(obj->size, bin_size, bins, num_bins);
        t1 = clock();
        my_tabulate(indices, num_bins, bin_relu);
        tab_time += (clock() - t1);
        t1 = clock();
        int fit_idx = my_min(indices, num_bins);
        reduce_time += (clock() - t1);

        if (thread_id==0) {
            if(fit_idx < INT_MAX){
                found_fit_flag = true;
                dev_bin *bin = &bins[fit_idx];
                bin->occupancy += obj->size;
                bin->obj_list.push_back(*obj);

                check_bin(bin, bin_size, __LINE__);
            }

            // If you don't find any, make a new bin
            if (!found_fit_flag) {
                if (num_bins >= maxsize - 1) {
                    *dev_retval_pt = -1;
                    return;
                }

                // Make a new bin and put the object in it
                dev_bin b = dev_bin(0);
                b.occupancy = obj->size;
                b.obj_list.push_back(*obj);

                // Add this bin to bins
                bins[num_bins] = b;
                num_bins++;
                globals.num_bins = num_bins;

            }
        }
        __syncthreads();
    }
    if (thread_id == 0) {
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
          check_bin(&bins[j], bin_size, __LINE__);
        }

        // Return the number of bins
        *dev_retval_pt = (int) num_bins;
        printf("Finished, Num bins: %i\n", num_bins);
        long long total = (clock() - start);
        printf("sorting: %lf\n", sort_time / total);
        printf("tabing: %lf\n", tab_time / total);
        printf("reducing: %lf\n", reduce_time / total);
        delete[] indices;

        globals.clear();
    }

    return;
}


__global__
void kernelWalkPack() {
    // ID of thread within this trial. Trials are mapped to blocks,
    //  so thread_id does not depend on blockIdx
    int thread_id = threadIdx.x;
    int trial_id = blockIdx.x;

    // Read params
    int bin_size = params.bin_size;
    int maxsize = params.maxsize;
    volatile size_idx_pair *trial_sizes = params.trial_sizes;
    int *dev_retval_pt = params.dev_retval_pt;
    obj *obj_out = params.obj_out;
    size_t *idx_out = params.idx_out;

    // Every trial has its own globals, num_bins, and bins array
    __shared__ size_t num_bins;
    __shared__ dev_bin *bins;

    if(thread_id == 0){
        globals = cudaGlobals(maxsize, trial_id);
        num_bins = 0;
        bins = globals.bins;
    }

    __syncthreads();

    // Start with next fit
    if(thread_id == 0){
        runNF();
        num_bins = globals.num_bins;
    }


    if(num_bins >= maxsize){
        *dev_retval_pt = -1;
        if(thread_id == 0)
            printf("Next Fit failed\n");

        return;
    }

    __syncthreads();

    // This represents just one trial
    const int passes = 400;
    for(int pass = 0; pass < passes; pass++){
        // Optimize
        if (num_bins <= 1) {
            printf("Optimized to %d bins at the start of pass %d, breaking\n",
                   num_bins, pass);
            break;
        }
        if(thread_id == 0){
            size_t src;
            setup_rand();
            src = rand_full();

            dev_bin *srcbin = &bins[src];
            for(uint32_t i = 0; i < srcbin->obj_list.size(); i++){
                uint32_t obj_size = srcbin->obj_list.arr[i].size;

                // Choose a destination bin that is not the src and has enough space
                size_t dest;
                const int retries = 1000; // How many destinations to try

                for(int j = 0; j < retries; j++){
                    // Choose a destination other than the src
                    while(src == (dest = rand_empty()));

                    if(bins[dest].occupancy + obj_size <= bin_size){
                        break;
                    }
                }
                dev_bin *destbin = &bins[dest];

                destbin->obj_list.push_back(srcbin->obj_list.arr[i]);
                destbin->occupancy += obj_size;
            }
            // Delete srcbin
            bins[src].clear();
            for(size_t i = src; i < num_bins; i++){
                bins[i] = bins[i+1];
            }
            num_bins--;
            globals.num_bins = num_bins;
        }
        __syncthreads();
        size_t old_num_bins = num_bins; // Freeze num_bins for the for loop

        // Constrain all the old bins
        // All new bins must not be overfull
        if(thread_id == 0){
            for (size_t i = 0; i < old_num_bins; i++) {
                dev_bin *bin = &bins[i];
                dev_bin *newbin;

                // If bin is overfull, allocate a new bin
                while (bin->occupancy > bin_size) {

                    // Requires atomic add, but very low contention
                    size_t new_bin_idx =
                        atomicAdd((unsigned long long int *)(&num_bins), 1);

                    if (new_bin_idx >= maxsize) {
                        *dev_retval_pt = -1;
                        printf("Too many bins\n");
                        return;
                    }

                    // All bins are already zero-initialized
                    newbin = &bins[new_bin_idx];

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
    }

    printf("reducing\n");

    // Parallel reduction across blocks
    if(thread_id == 0){
        // Level 0: all threads write num_bins to [0, TRIALS)
        trial_sizes[trial_id].num_bins = num_bins;
        trial_sizes[trial_id].index = trial_id;

        int base_idx = 0;
        int level = 1;
        size_t min_bins = num_bins;
        int min_idx = trial_id;

        while((1 << level) <= TRIALS){
            // Wait for paired thread to submit a value
            int my_idx = base_idx + (trial_id >> (level - 1));
            int pair_idx = my_idx ^ 0x1;
            while(trial_sizes[pair_idx].num_bins == 0);

            if(trial_sizes[pair_idx].num_bins < min_bins ||
               (trial_sizes[pair_idx].num_bins == min_bins && pair_idx < my_idx)){
                globals.clear();
                return;
            }

            // Write to new location
            base_idx += TRIALS >> (level - 1);
            trial_sizes[base_idx + trial_id / (1 << level)].num_bins = min_bins;
            trial_sizes[base_idx + trial_id / (1 << level)].index = min_idx;
            level++;
        }

        *(params.best_trial) = min_idx;

        // Copy objects in order to obj_out
        // idx_out holds the indices into obj_out where each bin starts

        printf("Trial %d | Outputting (%lu bins)\n", trial_id, num_bins);
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
            check_bin(&bins[j], bin_size, __LINE__);
        }

        // Return the number of bins
        *dev_retval_pt = num_bins;

        globals.clear();
    }
    return;
}

void setup(cudaParams& p) {
    int maxsize;
    int neg_one = -1;

    // Calculate a high water mark for number of bins used
    maxsize = calculate_maxsize();
    cout << "Max number of bins " << maxsize << std::endl;

    // Assign parameters for device
    p.total_obj_size = host_total_obj_size;
    p.bin_size = host_bin_size;
    p.num_objs = host_num_objs;
    p.maxsize = maxsize;

    cout << "allocating space" << std::endl;

    // Allocate space on device
    gpuErrchk(cudaMalloc(&p.dev_retval_pt, sizeof(int)));
    gpuErrchk(cudaMalloc(&p.objs, host_num_objs * sizeof(obj)));
    gpuErrchk(cudaMalloc(&p.obj_out, host_num_objs * sizeof(obj)));
    gpuErrchk(cudaMalloc(&p.idx_out, host_num_objs * sizeof(size_t)));
    gpuErrchk(cudaMalloc(&p.trial_sizes, 2 * TRIALS * sizeof(size_idx_pair)));
    gpuErrchk(cudaMalloc(&p.best_trial, sizeof(int)));
    cout << "copying inputs" << std::endl;
    gpuErrchk(cudaMemcpy(p.objs, host_objs, host_num_objs * sizeof(obj), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(params, &p, sizeof(cudaParams)));
    gpuErrchk(cudaMemset(p.trial_sizes, 0, 2 * TRIALS * sizeof(size_idx_pair)));
    gpuErrchk(cudaMemcpy(p.best_trial, &neg_one, sizeof(int), cudaMemcpyHostToDevice));
    cout << "setup done" << std::endl;
}

void cleanup(cudaParams& p) {
    // Outputs
    obj *obj_out;
    size_t *idx_out;
    int *dev_retval_pt, host_retval;
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

void runBFD(){
    //Inputs
    cudaParams p;

    setup(p);

    // Run BFD
    kernelBFD<<<1,dim3(threads_per_block, 1)>>>();
    cudaThreadSynchronize();

    cleanup(p);
}


__host__
void run() {
    //Inputs
    cudaParams p;

    setup(p);

    // Run WalkPack
    int threads_per_trial = 1;
    dim3 walkpack_block_dim(threads_per_trial, 1);
    dim3 walkpack_grid_dim (TRIALS, 1);

    cout << "Running kernel" << std::endl;
    kernelWalkPack<<<walkpack_grid_dim, walkpack_block_dim>>>();
    cudaThreadSynchronize();
    cout << "Kernel done" << std::endl;
    cleanup(p);
}