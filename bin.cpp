#include <assert.h>
#include <fstream>
#include <random>
#include <algorithm>

#include "bin.h"
#include "parse.h"

//#define DEBUG
#ifdef DEBUG
    #define DEBUG_CHECK_BIN(bin) check_bin(bin)
#else
    #define DEBUG_CHECK_BIN(bin)
#endif

using namespace std;

uint32_t total_obj_size = 0; //pseudo-constant
uint32_t bin_size;
uint32_t num_objs;
obj_t *objs;
vector<bin_t> bins;
vector<bin_t> best_bins;
vector<float> ecdfs; // CDF of empty space
vector<float> fcdfs; // CDF of full space
// For int-based version
//vector<uint32_t> fcdfs; // CDF of full space
uint32_t fcdf_max;
alias *alias_table;

uint32_t host_num_bins;
bin *bins_out;

// Allocate and populate a bin_t with the objs in obj_list
bin_t *make_bin(vector<obj_t> obj_list){
    bin_t *b = new bin_t();
    b->obj_list = obj_list;

    uint32_t occupancy = 0;
    for(size_t i = 0; i < obj_list.size(); i++){
        occupancy += obj_list[i].size;
    }
    b->occupancy = occupancy;

    return b;
}

// Allocate and populate a bin_t with only obj
bin_t *make_bin(obj_t *obj){
    bin_t *b = new bin_t();
    b->obj_list.push_back(*obj);
    b->occupancy = obj->size;

    return b;
}

void check_bin(bin_t *b) {
    uint32_t sum = 0;
    for (obj_t obj : b->obj_list) {
        sum += obj.size;
    }
    assert(b->occupancy == sum);
    assert(0 <= b->occupancy);
    assert(b->occupancy <= bin_size);
}

void constrain() {
    // Do want i to track the growing bins list here
    for (size_t i = 0; i < bins.size(); i++) {
        bin_t *bin = &bins[i];
        bin_t *b;
        bool new_bin_flag = false;
        if (bin->occupancy > bin_size) {
            b = new bin_t();
            new_bin_flag = true;
        }
        while (bin->occupancy > bin_size) {
            size_t r = rand() % bin->obj_list.size();
            obj_t obj = bin->obj_list[r];
            bin->obj_list.erase(bin->obj_list.begin() + r);
            bin->occupancy -= obj.size;
            b->occupancy += obj.size;
            b->obj_list.push_back(obj);
        }
        if (new_bin_flag) {
            bins.push_back(*b);
            DEBUG_CHECK_BIN(&bins[i]);
        }
    }

    for(size_t ii = 0; ii < bins.size(); ii++){
        DEBUG_CHECK_BIN(&bins[ii]);
    }

}

// Constrain a single bin
void constrain_bin(uint32_t idx) {
    bin_t *bin = &bins[idx];

    if (bin->occupancy > bin_size) {
        bin_t *newbin = new bin_t();

        while (bin->occupancy > bin_size) {
            size_t r = rand() % bin->obj_list.size();
            obj_t obj = bin->obj_list[r];
            bin->obj_list.erase(bin->obj_list.begin() + r);
            bin->occupancy -= obj.size;
            newbin->occupancy += obj.size;
            newbin->obj_list.push_back(obj);
        }
        DEBUG_CHECK_BIN(&bins[idx]);

        bins.push_back(*newbin);
        // If the new bin is also too full, fix it
        if(newbin->occupancy > bin_size){
            constrain_bin(bins.size() - 1);
        }
    }

    return;
}

// Recalculate data structures used by rand_empty & _full based on current bins.
void setup_rand(){
    ecdfs.resize(bins.size());
    fcdfs.resize(bins.size());

    float sum_empty_space = (float)(bins.size() * bin_size - total_obj_size);
    float ecdf = 0.f;
    float fcdf = 0.f;
    size_t i;
    for(i = 0; i < bins.size() - 1; i++){
        ecdf += ((float) (bin_size - bins[i].occupancy)) / sum_empty_space;
        ecdfs[i] = ecdf;
        fcdf += ((float) bins[i].occupancy) / total_obj_size;
        fcdfs[i] = fcdf;
    }
    // Make the final value (bins.size() - 1) greater than 1
    //  so upper_bound doesn't break
    ecdfs[i] = 2.f;
    fcdfs[i] = 2.f;
}

// Recalculate data structures used by rand_empty on current bins.
void setup_rand_empty(){
    ecdfs.resize(bins.size());

    float sum_empty_space = (float)(bins.size() * bin_size - total_obj_size);
    float ecdf = 0.f;
    size_t i;
    for(i = 0; i < bins.size() - 1; i++){
        ecdf += ((float) (bin_size - bins[i].occupancy)) / sum_empty_space;
        ecdfs[i] = ecdf;
    }
    // Make the final value greater than 1 so upper_bound doesn't break
    ecdfs[i] = 2.f;
}

// Recalculate data structures used by rand_full on current bins. Assign zero
//  probability to bins that can't fit src_occ;
void setup_rand_full(uint32_t src_occ, uint32_t src){
    fcdfs.resize(bins.size());

    uint32_t max_occ = bin_size - src_occ;

    uint32_t fcdf = 0;
    size_t i;
    for(i = 0; i < bins.size(); i++){
        // If a bin is too full, give it 0 probability
        if(bins[i].occupancy <= max_occ && i != src){
            fcdf += bins[i].occupancy;
        }

        fcdfs[i] = fcdf;
    }

    fcdf_max = fcdf;
}

// Select a random bin weighted in favor of empty bins. Uses data structures
//  generated in setup_rand, which may be stale.
uint32_t rand_empty(){
    size_t ret = upper_bound(ecdfs.begin(), ecdfs.end(),
                             ((float)rand()) / RAND_MAX) - ecdfs.begin();
    assert(0 <= ret);
    assert(ret < bins.size());
    return ret;
}

// Select a random bin weighted in favor of full bins.
uint32_t rand_full(){
    size_t ret = upper_bound(fcdfs.begin(), fcdfs.end(), ((float)rand()) / RAND_MAX) -
             fcdfs.begin();
    assert(0 <= ret);
    assert(ret < bins.size());
    return ret;
}
/* int-based version
uint32_t rand_full(){
    size_t ret;
    if(fcdf_max == 0) {
        ret = rand() % bins.size();
    } else {
        ret = upper_bound(fcdfs.begin(), fcdfs.end(), rand() % fcdf_max)
                      - fcdfs.begin();
    }
    assert(0 <= ret);
    assert(ret < bins.size());
    return ret;
}
*/

int overflow_count = 0;
int trial_count = 0;
void optimize() {
    if (bins.size() <= 1) {
        return;
    }

    setup_rand();
    size_t src = rand_empty();
    bin_t *srcbin = &bins[src];

    for(uint32_t i = 0; i < srcbin->obj_list.size(); i++){
        uint32_t obj_size = srcbin->obj_list[i].size;

        // Choose a destination bin that is not the src and has enough space
        size_t dest;
        const int retries = 1000; // How many destinations to try
        trial_count++;
        overflow_count++;
        for(int j = 0; j < retries; j++){
            // Choose a destination other than the src
            while(src == (dest = rand_full()));

            if(bins[dest].occupancy + obj_size <= bin_size){
                overflow_count--;
                break;
            }
        }
        bin_t *destbin = &bins[dest];

        destbin->obj_list.push_back(srcbin->obj_list[i]);
        destbin->occupancy += obj_size;
    }

    // Delete srcbin
    bins.erase(bins.begin() + src);

    // TODO: only constrain the bins that need constraining
    // if(src < dest){
    //     dest--;
    // }

    // if(bins[dest].occupancy > bin_size){
    //     overflow_count++;
    //     constrain_bin(dest);
    // }
    constrain();
}

// Fill bins using best-fit decreasing
void runBFD() {
    sort(objs, &objs[num_objs],
        [](const obj_t &a, const obj_t &b) -> bool { return a.size > b.size; });

    for (size_t i = 0; i < num_objs; i++) {
        obj_t obj = objs[i];
        bool found_fit_flag = false;
        for (size_t j = 0; j < bins.size(); j++) {
            bin_t *bin = &bins[j];
            if (bin->occupancy + obj.size <= bin_size) {
                bin->occupancy += obj.size;
                bin->obj_list.push_back(obj);
                found_fit_flag = true;
                break;
            }
        }
        if (!found_fit_flag) {
            bins.push_back(*make_bin(&obj));
        }
    }

    return;
}

// Fill bins using next-fit
void runNF() {
    bins.clear();
    bin_t *b = new bin_t();

    for (size_t i = 0; i < num_objs; i++) {
        obj_t *o = &objs[i];
        if(b->occupancy + o->size > bin_size){
            bins.push_back(*b);
            b = new bin_t();
        }
        b->occupancy += o->size;
        b->obj_list.push_back(*o);
    }
    bins.push_back(*b);

    return;
}

void run() {
    // bins.push_back(*make_bin(&objs[0]));
    // for (size_t i = 1; i < num_objs; i++) {
    //     obj_t obj = objs[i];
    //     bins[0].obj_list.push_back(obj);
    //     bins[0].occupancy += obj.size;
    // }
    // constrain();

    objs = host_objs;
    num_objs = host_num_objs;
    bin_size = host_bin_size;
    total_obj_size = host_total_obj_size;

    runNF();
    srand(123412341);
    const int bins_per_pass = 1;
    const int passes = 1000;
    const int trials = 50;
    uint32_t best_size = UINT32_MAX;
    vector<bin_t> seed = bins;
    for (int trial = 0; trial < trials; trial++) {
        for (int i = 0; i < passes; i++) {
            for (int j = 0; j < bins_per_pass; j++) {
                optimize();
                for(size_t ii = 0; ii < bins.size(); ii++){
                    DEBUG_CHECK_BIN(&bins[ii]);
                }
            }
        }

        if (bins.size() < best_size) {
            printf("Size %d\n", (int)bins.size());
            best_bins = bins;
            best_size = bins.size();
        }
        bins = seed;
    }
    bins = best_bins;

    bins_out = &bins[0];
    host_num_bins = bins.size();

    printf("Overflow: %d / %d\n", overflow_count, trial_count);

    return;
}

