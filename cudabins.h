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
    }

    __device__
    void push_back (const T& val) {
        if (num_entries + 1 >= maxlen) {
            T *old = arr;
            arr = new T[maxlen * SCALE];
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
        for (size_t i = 0; i < copy_size; i++) {
            arr[i] = old[i];
        }
        delete[] old;

    }
    __device__
    T &operator[] (int i) {
        return arr[i];
    }
    __device__
    uint32_t upper_bound (T target) {
        for (uint32_t i = 0; i < num_entries; i++) {
            if (arr[i] > target) {
                return i;
            }
        }
        return num_entries - 1;
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
    __device__
    dev_bin() {}
    __device__
    dev_bin(int dummy) {
        occupancy = 0;
        obj_list = ghetto_vec<obj> (0);
    }
};

struct bin {
  uint32_t occupancy;
  thrust::host_vector<obj> obj_list;
  alias alias;
  __host__ ~bin() {}
};

struct cudaParams {
    uint32_t total_obj_size;
    uint32_t bin_size;
    uint32_t num_objs;
    int maxsize;
    int *dev_retval_pt;
    obj *objs;
    obj *obj_out;
    size_t *idx_out;
};

struct cudaGlobals {
    dev_bin *bins;
    ghetto_vec<float> ecdfs;
    ghetto_vec<float> fcdfs;
    size_t size;
    curandState s;

    __device__
    cudaGlobals() {}
    __device__
    cudaGlobals(int maxsize) {
        bins = (dev_bin *) malloc(sizeof(dev_bin) * maxsize);
        for (int i = 0; i < maxsize; i++) {
            bins[i] = dev_bin(0); //dummy variable in constructor
        }
        ecdfs = ghetto_vec<float> (0);
        fcdfs = ghetto_vec<float> (0);
        size = 0;
        curand_init(SEED, 0, 0, &s);
    }

    __device__
    float rand() {
        return curand_uniform(&s);
    }
};


#endif /* _CUDABINS_H_ */