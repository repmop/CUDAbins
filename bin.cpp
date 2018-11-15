#include "bin.h"
#include <json/json.h>
#include <assert.h>
#include <jsoncpp.cpp>
#include <iostream>
#include <fstream>
#include <random>

using namespace std;

uint32_t bin_size;
uint32_t num_objs;
obj_t *objs;
vector<bin_t> bins;

// Allocate and populate a bin_t with the objs in obj_list
bin_t *make_bin(vector<obj_t> obj_list){
    bin_t *b = new bin_t();
    b->capacity = bin_size;
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
    b->capacity = bin_size;
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
    assert(b->occupancy <= b->capacity);
}

void constrain() {
    // Do want i to track the growing bins list here
    for (size_t i = 0; i < bins.size(); i++) {
        bin_t *bin = &bins[i];
        bin_t *b;
        bool new_bin_flag = false;
        if (bin->occupancy > bin->capacity) {
            b = new bin_t();
            b->capacity = bin->capacity;
            new_bin_flag = true;
        }
        while (bin->occupancy > bin->capacity) {
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
}

void optimize() {
    if (bins.size() <= 1) {
        return;
    }
    size_t r = rand() % bins.size(); 
    bin_t *bin = &bins[r];
    while (bin->obj_list.size() > 0 && bins.size() > 1) {
        obj_t obj = bin->obj_list.back();
        bin->occupancy -= obj.size;
        size_t temp_r;
        while (r == (temp_r = rand() % bins.size()));
        bins[temp_r].occupancy += obj.size;
        bins[temp_r].obj_list.push_back(obj);
        bin->obj_list.pop_back();
    }
    bins.erase(bins.begin() + r);

}
// #define BFD
void run() {
    #ifdef BFD
    sort(objs, &objs[num_objs],
        [](const obj_t &a, const obj_t &b) -> bool { return a.size > b.size; });

    for (size_t i = 0; i < num_objs; i++) {
        obj_t obj = objs[i];
        bool found_fit_flag = false;
        for (size_t j = 0; j < bins.size(); j++) {
            bin_t *bin = &bins[j];
            if (bin->occupancy + obj.size <= bin->capacity) {
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
    #else
    bins.push_back(*make_bin(&objs[0]));
    for (size_t i = 1; i < num_objs; i++) {
        obj_t obj = objs[i];
        bins[0].obj_list.push_back(obj);
        bins[0].occupancy += obj.size;
    }
    constrain();
    const int bins_per_pass = 30;
    const int passes = 1000;
    for (int i = 0; i < passes; i++) {
        for (int j = 0; j < bins_per_pass; j++) {
            optimize();
        }
        constrain();
    }

    #endif
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
        cout << obj_data;
    } else { //print to file
        filebuf fb;
        fb.open(outfile, ios::out);
        ostream f(&fb);
        f << obj_data;
    }
    return true;
}
