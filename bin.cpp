#include "bin.h"
#include <json/json.h>
#include <jsoncpp.cpp>
#include <iostream>
#include <fstream>

using namespace std;

uint32_t bin_size;
uint32_t num_objs;
obj_t *objs;
vector<bin_t> *bins;

// Allocate and populate a bin_t with the objs in obj_list
bin_t *make_bin(vector<obj_t> *obj_list){
    bin_t *b = new bin_t();
    b->capacity = bin_size;
    b->obj_list = obj_list;

    uint32_t occupancy = 0;
    for(uint32_t i = 0; i < obj_list->size(); i++){
        occupancy += (*obj_list)[i].size;
    }
    b->occupancy = occupancy;

    return b;
}

// Allocate and populate a bin_t with only obj
bin_t *make_bin(obj_t *obj){
    bin_t *b = new bin_t();
    b->capacity = bin_size;
    b->obj_list = new vector<obj_t>(1); // TODO: find a good initial size
    (*(b->obj_list))[0] = *obj;
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

    // Initialize object array and put one obj in each bin
    objs = new obj_t[num_objs];
    bins = new vector<bin_t>(num_objs);
    auto obj_array = obj_data["objs"];
    for(uint32_t i = 0; i < num_objs; i++){
        #ifdef TAGGING
        objs[i].tag = i;
        #endif
        objs[i].size = obj_array[i].asUInt();

        // TODO: aaaaaaa struct copying is scary
        (*bins)[i] = *make_bin(&objs[i]);
    }


    return true;
}


void run() {
    for(uint32_t i = 0; i < num_objs; i++)
        cout << objs[i].size << "\n";

    for(bin_t bin : *bins)
        cout << (*bin.obj_list)[0].size << "\n";

    return;
}

bool dump(char *outfile) {


    return true;
}
