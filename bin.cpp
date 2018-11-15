#include "bin.h"
#include <json/json.h>
#include <jsoncpp.cpp>
#include <iostream>
#include <fstream>

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


void run() {
    // worst case # bins == # objs
    for(uint32_t i = 0; i < num_objs; i++) {

    }
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

    // for(size_t i = 0; i < bins.size(); i++) {
    //     cout << "bin " << i << ":\n";
    //     bin_t bin = bins[i];
    //     for (size_t j = 0; j < bin.obj_list.size(); j++) {
    //         cout << bin.obj_list[j].size << "\t";
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    // for(size_t i = 0; i < num_objs; i++) {
    //     cout << objs[i].size << endl;
    // }
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
