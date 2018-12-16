#ifndef _CUDABINS_H_
#define _CUDABINS_H_

#include <curand_kernel.h>

#define SCALE 2
#define SEED 1274
#define DEF_ENTRIES 10

template <typename T>
struct ghetto_vec {
    int num_entries;
    int maxlen;
    T *arr;
    __device__
    ghetto_vec() {}

    __device__
    ghetto_vec(int dummy) {
        num_entries = 0;
        maxlen = DEF_ENTRIES;
        arr = new T[maxlen];
        if(arr == NULL)
            printf("Failed to make new arr\n");
    }
    __device__
    void erase(uint32_t ind) {
        for (int i = ind; i < num_entries - 1; i++) {
            arr[i] = arr[i+1];
        }
        num_entries--;
    }

    __device__
    void push_back (const T& val) {
        if (num_entries + 1 >= maxlen) {
            T *old = arr;
            arr = new T[maxlen * SCALE];
            if(arr == NULL)
                printf("push_back: Failed to make new arr\n");

            for (int i = 0; i < num_entries; i++) {
                arr[i] = old[i];
            }
            delete[] old;
        }
        arr[num_entries++] = val;
    }
    __device__
    int size() {
        return num_entries;
    }
    __device__
    void resize(uint32_t new_size) {
        //Could free and copy if memory becomes a concern
        if (new_size<=maxlen) {
            return;
        }
        T *old = arr;
        uint32_t copy_size;
        if (new_size > num_entries) {
            copy_size = num_entries;
        } else {
            copy_size = new_size;
        }
        maxlen = new_size * SCALE;
        arr = new T[maxlen];
        if(arr == NULL)
            printf("resize: Failed to make new arr\n");

        for (size_t i = 0; i < copy_size; i++) {
            arr[i] = old[i];
        }
        delete[] old;

    }
    __device__
    T &operator[] (int i) {
        return arr[i];
    }

    // For debug
    __device__ uint32_t seq_upper_bound (T target) {
        for (uint32_t i = 0; i < num_entries; i++) {
            if (arr[i] > target) {
                return i;
            }
        }
        return num_entries - 1;
    }

    // Returns the first index i s.t. arr[i] > target
    __device__
    uint32_t upper_bound(T target){
        uint32_t lo = 0;               // arr[lo] < target
        uint32_t hi = num_entries - 1; // arr[hi] >= target
        uint32_t mid;
        if(num_entries==0)   { return 0;  }
        if(arr[lo] >= target){ return lo; }
        if(arr[hi] <  target){ return hi; }

        while(hi - lo > 1){
            mid = lo + (hi-lo)/2;
            if(arr[mid] < target){
                lo = mid;
            } else {
                hi = mid;
            }
        }

        return hi;
    }
    __device__
    void clear() {
        delete [] arr;
    }

};

typedef struct obj {
    uint32_t size;
}obj;

struct alias {
    int i1;
    int i2;
    float divider;
};


struct dev_bin {
    uint32_t occupancy;
    ghetto_vec<obj> obj_list;
    alias alias;
    bool valid;
    __device__
    dev_bin() {}
    __device__
    dev_bin(int dummy) {
        occupancy = 0;
        obj_list = ghetto_vec<obj> (0);
        valid = true;
    }
    __device__
    operator uint32_t () {
        return occupancy;
    }
    __device__
    void clear() {
        obj_list.clear();
    }
};

struct bin {
  uint32_t occupancy;
  std::vector<obj> obj_list;
  alias alias;
  __host__ ~bin() {}
};

struct size_idx_pair {
    size_t num_bins;
    int index;
};

struct cudaParams {
    // Inputs
    uint32_t total_obj_size;
    uint32_t bin_size;
    uint32_t num_objs;
    int maxsize;
    obj *objs;

    // Scratch
    size_idx_pair *trial_sizes;
    int *best_trial;

    // Outputs
    int *dev_retval_pt;
    obj *obj_out;
    size_t *idx_out;
};

struct cudaGlobals {
    dev_bin *bins;
    ghetto_vec<float> ecdfs;
    ghetto_vec<float> fcdfs;
    size_t num_bins;
    curandState s;
    int saved_maxsize;

    __device__
    cudaGlobals() {}
    __device__
    cudaGlobals(int maxsize, int seed = SEED) {
        bins = (dev_bin *) malloc(sizeof(dev_bin) * maxsize);
        if(bins == NULL) {
            printf("Failed to malloc bins\n");
            assert(false);
        }

        for (int i = 0; i < maxsize; i++) {
            //bins[i] = dev_bin(0); //dummy variable in constructor
        }
        ecdfs = ghetto_vec<float> (0);
        fcdfs = ghetto_vec<float> (0);
        num_bins = 0;
        curand_init(seed, 0, 0, &s);
        saved_maxsize = maxsize;
    }

    __device__
    float rand_f() {
        return ((float) curand(&s)) / ((float) UINT32_MAX);
    }
    __device__
    uint32_t rand_i() {
        return curand(&s);
    }

    __device__
    void clear() {
        for (int i = 0; i < saved_maxsize; i++) {
            bins[i].clear();
        }
        ecdfs.clear();
        fcdfs.clear();
        free(bins);
    }
};

struct fc_op : public thrust::binary_function<int,int,float>
{
    __device__
    fc_op(int dummy) {}
    __device__
    float operator()(int x, int y) {
        return ((float) x) + ((float) y);
    }
};

struct ec_op : public thrust::binary_function<int,int,float>
{
    int bin_size;
    __device__
    ec_op(int bs) {
        bin_size = bs;
    }
    __device__
    float operator()(int x, int y) {
        return ((float) x + (bin_size - y));
    }
};

struct div_op : public thrust::unary_function<float,float>
{
    int divend;
    __device__
    div_op(int div) {
        divend = div;
    }
    __device__
    float operator()(float x) {
        return x / divend;
    }
};

struct bin_relu_op : public thrust::unary_function<int,int>
{
    int target_size, max_size, num_bins;
    dev_bin *bins;
    __device__
    bin_relu_op(int target, int bin_size, dev_bin *dev_bins, size_t n) {
        target_size = target;
        max_size = bin_size;
        bins = dev_bins;
        num_bins = (int) n;
    }
    __device__
    int operator()(int i) {
        if(i < num_bins && bins[i].occupancy + target_size <= max_size){
            return i;
        } else {
            return INT_MAX;
        }
    }
};

/* Parsed Inputs */
extern obj *host_objs;
extern uint32_t host_num_objs;
extern uint32_t host_bin_size;
extern uint32_t host_total_obj_size; //pseudo-constant

#endif /* _CUDABINS_H_ */
