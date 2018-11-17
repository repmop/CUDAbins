#include "bin.h"
#include <json/json.h>
#include <assert.h>
#include <jsoncpp.cpp>
#include <iostream>
#include <fstream>
#include <random>

using namespace std;

uint32_t total_obj_size = 0; //pseudo-constant
uint32_t bin_size;
uint32_t num_objs;
obj_t *objs;
vector<bin_t> bins;
vector<float> ecdfs; // CDF of empty space
vector<float> fcdfs; // CDF of full space
alias *alias_table;

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

bool parse(char *infile) {
    ifstream f(infile, std::ifstream::binary);
    if (f.fail()) {
        return false;
    }
    Json::Value obj_data;
    f >> obj_data;
    bin_size = obj_data["bin_size"].asUInt();
    num_objs = obj_data["num_objs"].asUInt();
    // cout << obj_data     << endl;


    // Initialize object array and put one obj in each bin
    objs = new obj_t[num_objs];
    auto obj_array = obj_data["objs"];
    for(uint32_t i = 0; i < num_objs; i++){
        #ifdef TAGGING
        objs[i].tag = i;
        #endif
        objs[i].size = obj_array[i].asUInt();
        total_obj_size += obj_array[i].asUInt();

        // TODO: aaaaaaa struct copying is scary
        // bins.push_back(*make_bin(&objs[i]));
    }
    return true;
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
            check_bin(&bins[i]);
        }
    }

    for(size_t ii = 0; ii < bins.size(); ii++){
        check_bin(&bins[ii]);
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
        check_bin(&bins[idx]);

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
    for(uint32_t i = 0; i < bins.size(); i++){
        ecdf += ((float) (bin_size - bins[i].occupancy)) / sum_empty_space;
        ecdfs[i] = ecdf;
        fcdf += ((float) bins[i].occupancy) / total_obj_size;
        fcdfs[i] = fcdf;
    }
}

// Select a random bin weighted in favor of empty bins. Uses data structures
//  generated in setup_rand, which may be stale.
uint32_t rand_empty(){
    return upper_bound(ecdfs.begin(), ecdfs.end(), ((float)rand()) / RAND_MAX) -
             ecdfs.begin();
}

// Select a random bin weighted in favor of full bins.
uint32_t rand_full(){
    return upper_bound(fcdfs.begin(), fcdfs.end(), ((float)rand()) / RAND_MAX) -
             fcdfs.begin();
}

void optimize() {
    if (bins.size() <= 1) {
        return;
    }

    setup_rand();
    size_t src = rand_empty();
    // size_t src = rand() % bins.size();
    bin_t *srcbin = &bins[src];

    // TODO: consider changing how we break up srcbin to put each item in
    //  a different bin
    uint32_t src_occ = srcbin->occupancy;
    // Choose a destination bin that is not the src and has enough space
    size_t dest;
    int retries = 5; // How many destinations to try
    for(int i = 0; i < retries; i++){
        // Choose a destination other than the src
        while(src == (dest = rand_full()));

        if(bins[dest].occupancy + src_occ <= bin_size){
            break;
        }
    }
    bin_t *destbin = &bins[dest];

    check_bin(srcbin);
    check_bin(destbin);

    // Put src in dest
    destbin->obj_list.insert(destbin->obj_list.end(),
                             srcbin->obj_list.begin(), srcbin->obj_list.end());
    destbin->occupancy += srcbin->occupancy;

    // Delete srcbin
    bins.erase(bins.begin() + src);

    if(src < dest){
        dest--;
    }

    if(bins[dest].occupancy > bin_size){
        constrain_bin(dest);
    }

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

void run() {
    bins.push_back(*make_bin(&objs[0]));
    for (size_t i = 1; i < num_objs; i++) {
        obj_t obj = objs[i];
        bins[0].obj_list.push_back(obj);
        bins[0].occupancy += obj.size;
    }
    constrain();
    const int bins_per_pass = 1;
    const int passes = 0;
    for (int i = 0; i < passes; i++) {
        for (int j = 0; j < bins_per_pass; j++) {
            optimize();
            for(size_t ii = 0; ii < bins.size(); ii++){
                check_bin(&bins[ii]);
            }
        }
        //constrain();
    }

    return;
}

bool dump(char *outfile) {
    Json::Value obj_data;
    obj_data["bin_size"] = bin_size;
    obj_data["num_objs"] = num_objs;
    obj_data["num_bins"] = bins.size();
    obj_data["objs"] = Json::Value(Json::arrayValue);
    obj_data["bins"] = Json::Value(Json::arrayValue);
    for(uint32_t i = 0; i < num_objs; i++){
        obj_data["objs"][i] = objs[i].size;
    }
    for(uint32_t i = 0; i < (uint32_t) bins.size(); i++) {
        bin_t bin = bins[i];
        obj_data["bins"][i] = Json::Value(Json::arrayValue);
        for (uint32_t j = 0; j < bin.obj_list.size(); j++) {
            obj_data["bins"][i][j] = bin.obj_list[j].size;
        }
    }
    if (outfile==NULL) { //print results to stdout
        cout << "num_objs: " << num_objs << endl;
        cout << "num_bins: " << bins.size() << endl;
    } else { //print to file
        filebuf fb;
        fb.open(outfile, ios::out);
        ostream f(&fb);
        f << obj_data;
    }
    return true;
}
